#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

#include <vector>
#include <thread>

namespace py = pybind11;


// f, cx, cy

struct ReprojectionCost3 {
    ReprojectionCost3(double u, double v)
        : u(u), v(v) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const r,
                    const T* const t,
                    const T* const point_params,
                    const T* const dist_scale_params,
                    T* residuals) const {
        // Intrinsic parameters
        const T& f = camera[0];
        const T& cx = camera[1];
        const T& cy = camera[2];

        // 3D point
        const T* X = point_params;

        // Transform point to camera frame
        T dist_scale = dist_scale_params[0];
        // dist_scale = ceres::exp(dist_scale);
        dist_scale = dist_scale > 0.0 ? dist_scale+1.0 : ceres::exp(dist_scale);
        T X_cam[3];
        ceres::AngleAxisRotatePoint(r, X, X_cam);
        for (int i = 0; i < 3; ++i) {
            X_cam[i] += t[i];
            X_cam[i] *= dist_scale;
        }

        // Compute residuals
        T vdir[2] = { (u-cx)/f, (v-cy)/f };
        T inv_vdir_norm = 1.0 / ceres::sqrt(vdir[0]*vdir[0] + vdir[1]*vdir[1] + 1.0);
        // double inv_vdir_norm = 1.0;
        residuals[0] = X_cam[0] - inv_vdir_norm * vdir[0];
        residuals[1] = X_cam[1] - inv_vdir_norm * vdir[1];
        residuals[2] = X_cam[2] - inv_vdir_norm;

        return true;
    }

    double u;
    double v;
};

Eigen::MatrixXd solve_ba_3(
    Eigen::Ref<Eigen::VectorXd> camera,
    Eigen::Ref<Eigen::MatrixXd> poses,
    Eigen::Ref<Eigen::MatrixXd> points,
    Eigen::VectorXd &dist_scales,
    const Eigen::VectorXi &poses_i,
    const Eigen::VectorXi &points_i,
    const Eigen::MatrixXd &points2d,
    bool fixed_intrinsic,
    bool fixed_rotation,
    bool verbose
) {
    size_t n_pose = poses.rows();
    size_t n_point = points.rows();
    size_t n_obs = points2d.rows();
    assert(poses_i.size() == n_obs);
    assert(points_i.size() == n_obs);
    // std::cout << n_pose << " " << n_point << " " << n_obs << std::endl;

    ceres::Problem problem;

    // Camera intrinsics
    problem.AddParameterBlock((double*)camera.data(), 3);
    if (fixed_intrinsic)
        problem.SetParameterBlockConstant((double*)camera.data());

    // Poses
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    std::vector<Vector6d> poses_v(n_pose, Vector6d::Zero());
    for (size_t i = 0; i < n_pose; ++i) {
        poses_v[i] = poses.row(i);
        if (fixed_rotation) {
            problem.AddParameterBlock(poses_v[i].data(), 3);
            problem.SetParameterBlockConstant(poses_v[i].data());
        }
    }

    // Points
    std::vector<Eigen::Vector3d> points_v(n_point, Eigen::Vector3d::Zero());
    for (size_t i = 0; i < n_point; ++i)
        points_v[i] = points.row(i);

    // Distance scales
    std::vector<double> dist_scales_v(n_obs, 0.0);
    for (size_t i = 0; i < n_obs; ++i) {
        dist_scales_v[i] = dist_scales(i);
        // problem.AddParameterBlock(&dist_scales_v[i], 1);
        // problem.SetParameterLowerBound(&dist_scales_v[i], 0, 0.0);
    }

    // Residuals
    std::vector<ceres::ResidualBlockId> residual_blocks;
    for (size_t i = 0; i < n_obs; ++i) {
        int pose_idx = poses_i[i];
        int point_idx = points_i[i];
        assert(pose_idx >= 0 && pose_idx < n_pose);
        assert(point_idx >= 0 && point_idx < n_point);

        const Eigen::Vector2d& uv = points2d.row(i);

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionCost3, 3, 3, 3, 3, 3, 1>(
                new ReprojectionCost3(uv(0), uv(1)));

        ceres::LossFunction* loss_function =
            new ceres::HuberLoss(2.0 / hypot(960.0, 540.0));

        ceres::ResidualBlockId residual = problem.AddResidualBlock(
            cost_function, loss_function,
            (double*)camera.data(),
            (double*)poses_v[pose_idx].data(),
            (double*)poses_v[pose_idx].data()+3,
            (double*)points_v[point_idx].data(),
            &dist_scales_v[i]);
        
        residual_blocks.push_back(residual);
    }

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = verbose;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-6;
    options.parameter_tolerance = 1e-5;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = std::thread::hardware_concurrency();

    // Run the solver
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Print initial and final cost
    std::cout << "cost: " << (summary.initial_cost/n_obs) <<
            " -> " << (summary.final_cost/n_obs) << std::endl;

    // Update results
    for (size_t i = 0; i < n_pose; ++i)
        poses.row(i) = poses_v[i];
    for (size_t i = 0; i < n_point; ++i)
        points.row(i) = points_v[i];
    for (size_t i = 0; i < n_obs; ++i)
        dist_scales(i) = dist_scales_v[i];

    // Get residuals
    Eigen::MatrixXd residuals(n_obs, 3);
    std::vector<double> evaluated_residuals;
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.residual_blocks = residual_blocks;
    problem.Evaluate(eval_options, nullptr, &evaluated_residuals, nullptr, nullptr);
    for (int i = 0; i < n_obs; i++)
        residuals.row(i) << evaluated_residuals[3*i], evaluated_residuals[3*i+1], evaluated_residuals[3*i+2];
    return residuals;
}


struct HierarchyReprojectionCost3 {
    HierarchyReprojectionCost3(int n_pose, double u, double v)
        : n_pose(n_pose), u(u), v(v) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        const T* const camera = parameters[0];
        const T* const point_params = parameters[1];
        const T* const dist_scale_params = parameters[2];

        // Intrinsic parameters
        const T& f = camera[0];
        const T& cx = camera[1];
        const T& cy = camera[2];

        // 3D point
        T X[3] = { point_params[0], point_params[1], point_params[2] };

        // Rotation and translation
        T X_cam[3];
        for (int pi = 0; pi < n_pose; pi++) {
            const T* const pose_params = parameters[pi+3];
            const T* r = pose_params;
            const T* t = pose_params + 3;
            ceres::AngleAxisRotatePoint(r, X, X_cam);
            for (int i = 0; i < 3; ++i) {
                X[i] = X_cam[i] + t[i];
            }
        }

        // Scaling
        T dist_scale = dist_scale_params[0];
        // dist_scale = ceres::exp(dist_scale);
        dist_scale = dist_scale > 0.0 ? dist_scale+1.0 : ceres::exp(dist_scale);
        for (int i = 0; i < 3; ++i) {
            X_cam[i] = X[i] * dist_scale;
        }

        // Compute residuals
        T vdir[2] = { (u-cx)/f, (v-cy)/f };
        T inv_vdir_norm = 1.0 / ceres::sqrt(vdir[0]*vdir[0] + vdir[1]*vdir[1] + 1.0);
        // double inv_vdir_norm = 1.0;
        residuals[0] = X_cam[0] - inv_vdir_norm * vdir[0];
        residuals[1] = X_cam[1] - inv_vdir_norm * vdir[1];
        residuals[2] = X_cam[2] - inv_vdir_norm;

        return true;
    }

    int n_pose;
    double u;
    double v;
};

struct HierarchyPoseRegularizationCost {
    HierarchyPoseRegularizationCost(int n_pose, double eps=1.0)
        : n_pose(n_pose), eps(eps) {}

    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const {
        for (int i = 0; i < 6; i++)
            residuals[i] = T(0.0);

        for (int pi = 0; pi < n_pose; pi++) {
            const T* const pose = parameters[pi];
            for (int i = 0; i < 6; i++)
                residuals[i] += pose[i];
        }

        for (int i = 0; i < 6; i++)
            residuals[i] *= eps;

        return true;
    }

    int n_pose;
    double eps;
};

Eigen::MatrixXd solve_ba_3_hierarchy(
    Eigen::Ref<Eigen::VectorXd> camera,
    Eigen::Ref<Eigen::MatrixXd> poses,
    Eigen::Ref<Eigen::MatrixXd> points,
    Eigen::VectorXd &dist_scales,
    const Eigen::VectorXi &poses_psa,
    const Eigen::VectorXi &poses_indices,
    const Eigen::VectorXi &poses_i,
    const Eigen::VectorXi &points_i,
    const Eigen::MatrixXd &points2d,
    const Eigen::VectorXi &pose_reg_psa,
    const Eigen::VectorXi &pose_reg_indices,
    bool fixed_intrinsic,
    bool verbose
) {
    size_t n_pose = poses.rows();
    size_t n_point = points.rows();
    size_t n_obs = points2d.rows();
    assert(poses_i.size() == n_obs);
    assert(points_i.size() == n_obs);
    // std::cout << n_pose << " " << n_point << " " << n_obs << std::endl;

    ceres::Problem problem;

    // Camera intrinsics
    problem.AddParameterBlock((double*)camera.data(), 3);
    if (fixed_intrinsic)
        problem.SetParameterBlockConstant((double*)camera.data());

    // Poses
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    std::vector<Vector6d> poses_v(n_pose, Vector6d::Zero());
    for (size_t i = 0; i < n_pose; ++i)
        poses_v[i] = poses.row(i);

    // Points
    std::vector<Eigen::Vector3d> points_v(n_point, Eigen::Vector3d::Zero());
    for (size_t i = 0; i < n_point; ++i)
        points_v[i] = points.row(i);

    // Distance scales
    std::vector<double> dist_scales_v(n_obs, 0.0);
    for (size_t i = 0; i < n_obs; ++i) {
        dist_scales_v[i] = dist_scales(i);
        // problem.AddParameterBlock(&dist_scales_v[i], 1);
        // problem.SetParameterLowerBound(&dist_scales_v[i], 0, 0.0);
    }

    const double eps = 1.0 / hypot(960.0, 540.0);

    // Residuals
    std::vector<ceres::ResidualBlockId> residual_blocks;
    for (size_t i = 0; i < n_obs; ++i) {
        int pose_idx = poses_i[i];
        int point_idx = points_i[i];
        assert(pose_idx >= 0 && pose_idx < n_pose);
        assert(point_idx >= 0 && point_idx < n_point);

        int pi0 = poses_psa[pose_idx];
        int pi1 = poses_psa[pose_idx+1];

        const Eigen::Vector2d& uv = points2d.row(i);

        ceres::DynamicCostFunction* cost_function =
            new ceres::DynamicAutoDiffCostFunction<HierarchyReprojectionCost3>(
                new HierarchyReprojectionCost3(pi1-pi0, uv(0), uv(1)));
        std::vector<double*> parameter_blocks = {
            (double*)camera.data(),
            (double*)points_v[point_idx].data(),
            &dist_scales_v[i],
        };
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(3);
        cost_function->AddParameterBlock(1);
        for (int pi = pi0; pi < pi1; pi++) {
            cost_function->AddParameterBlock(6);
            int pri = poses_indices[pi];
            parameter_blocks.push_back((double*)poses_v[pri].data());
        }
        cost_function->SetNumResiduals(3);

        ceres::LossFunction* loss_function =
            new ceres::HuberLoss(2.0 * eps);

        ceres::ResidualBlockId residual = problem.AddResidualBlock(
            cost_function, loss_function, parameter_blocks);

        residual_blocks.push_back(residual);
    }

    // Regularizations
    for (size_t i = 1; i < pose_reg_psa.rows(); ++i) {
        int ri0 = pose_reg_psa[i-1];
        int ri1 = pose_reg_psa[i];

        ceres::DynamicCostFunction* cost_function =
            new ceres::DynamicAutoDiffCostFunction<HierarchyPoseRegularizationCost>(
                new HierarchyPoseRegularizationCost(ri1-ri0, eps));
        std::vector<double*> parameter_blocks;
        for (int ri = ri0; ri < ri1; ri++) {
            cost_function->AddParameterBlock(6);
            int pri = pose_reg_indices[ri];
            parameter_blocks.push_back((double*)poses_v[pri].data());
        }
        cost_function->SetNumResiduals(6);

        ceres::ResidualBlockId residual = problem.AddResidualBlock(
            cost_function, nullptr, parameter_blocks);

        residual_blocks.push_back(residual);
    }

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = verbose;
    options.max_num_iterations = 200;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-6;
    options.parameter_tolerance = 1e-5;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = std::thread::hardware_concurrency();

    // Run the solver
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Print initial and final cost
    std::cout << "cost: " << (summary.initial_cost/n_obs) <<
            " -> " << (summary.final_cost/n_obs) << std::endl;

    // Update results
    for (size_t i = 0; i < n_pose; ++i)
        poses.row(i) = poses_v[i];
    for (size_t i = 0; i < n_point; ++i)
        points.row(i) = points_v[i];
    for (size_t i = 0; i < n_obs; ++i)
        dist_scales(i) = dist_scales_v[i];

    // Get residuals
    Eigen::MatrixXd residuals(n_obs, 3);
    std::vector<double> evaluated_residuals;
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.residual_blocks = residual_blocks;
    problem.Evaluate(eval_options, nullptr, &evaluated_residuals, nullptr, nullptr);
    for (int i = 0; i < n_obs; i++)
        residuals.row(i) << evaluated_residuals[3*i], evaluated_residuals[3*i+1], evaluated_residuals[3*i+2];
    return residuals;
}


PYBIND11_MODULE(ba_solver, m) {
    m.def("solve_ba_3", &solve_ba_3, "");
    m.def("solve_ba_3_hierarchy", &solve_ba_3_hierarchy, "");
}
