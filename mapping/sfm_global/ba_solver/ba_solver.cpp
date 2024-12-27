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
                    const T* const pose_params,
                    const T* const point_params,
                    const T* const dist_scale_params,
                    T* residuals) const {
        // Intrinsic parameters
        const T& f = camera[0];
        const T& cx = camera[1];
        const T& cy = camera[2];

        // Pose parameters
        const T* r = pose_params;
        const T* t = pose_params + 3;

        // 3D point
        const T* X = point_params;

        // Transform point to camera frame
        T dist_scale = ceres::exp(dist_scale_params[0]);
        T X_cam[3];
        ceres::AngleAxisRotatePoint(r, X, X_cam);
        for (int i = 0; i < 3; ++i) {
            X_cam[i] += t[i];
            X_cam[i] *= dist_scale;
        }

        // Compute residuals
        residuals[0] = X_cam[0] - (u-cx)/f;
        residuals[1] = X_cam[1] - (v-cy)/f;
        residuals[2] = X_cam[2] - 1.0;

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
    for (size_t i = 0; i < n_obs; ++i)
        dist_scales_v[i] = dist_scales(i);

    // Residuals
    std::vector<ceres::ResidualBlockId> residual_blocks;
    for (size_t i = 0; i < n_obs; ++i) {
        int pose_idx = poses_i[i];
        int point_idx = points_i[i];
        assert(pose_idx >= 0 && pose_idx < n_pose);
        assert(point_idx >= 0 && point_idx < n_point);

        const Eigen::Vector2d& uv = points2d.row(i);

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionCost3, 3, 3, 6, 3, 1>(
                new ReprojectionCost3(uv(0), uv(1)));

        ceres::LossFunction* loss_function =
            new ceres::HuberLoss(2.0 / hypot(960.0, 540.0));

        ceres::ResidualBlockId residual = problem.AddResidualBlock(
            cost_function, loss_function,
            (double*)camera.data(),
            (double*)poses_v[pose_idx].data(),
            (double*)points_v[point_idx].data(),
            &dist_scales_v[i]);
        
        residual_blocks.push_back(residual);
    }

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = verbose;
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
}
