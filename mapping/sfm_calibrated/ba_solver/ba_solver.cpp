#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <ceres/ceres.h>
#include <Eigen/Core>

#include <vector>
#include <thread>

namespace py = pybind11;

// Helper function for exponential map of SO(3)
Eigen::Matrix3d so3_exp(Eigen::Vector3d w) {
    double theta = w.norm();
    Eigen::Vector3d n = w / theta;
    Eigen::Matrix3d n_hat;
    n_hat << 0.0, -n(2), n(1),
             n(2), 0.0, -n(0),
             -n(1), n(0), 0.0;
    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity() + n_hat;
    } else {
        return cos(theta) * Eigen::Matrix3d::Identity() +
            (1.0 - cos(theta)) * n * n.transpose() +
            sin(theta) * n_hat;
    }
}


// fx, fy, cx, cy, k1, k2, p1, p2

struct ReprojectionCost8 {
    ReprojectionCost8(double u, double v)
        : u(u), v(v) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const pose_params,
                    const T* const point_params,
                    T* residuals) const {
        // Intrinsic parameters
        const T& fx = camera[0];
        const T& fy = camera[1];
        const T& cx = camera[2];
        const T& cy = camera[3];
        const T& k1 = camera[4];
        const T& k2 = camera[5];
        const T& p1 = camera[6];
        const T& p2 = camera[7];

        // Pose parameters
        const T* r = pose_params;
        const T* t = pose_params + 3;

        // 3D point
        const T* X = point_params;

        // so3 exponentiation
        T theta = ceres::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2] + T(1e-12));
        T n[3] = { r[0]/theta, r[1]/theta, r[2]/theta };
        T R[3][3] = {
            { T(0.0), -n[2], n[1] },
            { n[2], T(0.0), -n[0] },
            { -n[1], n[0], T(0.0) }
        };
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R[i][j] = ceres::cos(theta) * T(i==j ? 1.0 : 0.0) +
                    (T(1.0) - ceres::cos(theta)) * n[i] * n[j] +
                    ceres::sin(theta) * R[i][j];
            }
        }

        // Compute projection
        T X_cam[3];
        for (int i = 0; i < 3; ++i) {
            X_cam[i] = T(0);
            for (int j = 0; j < 3; ++j) {
                X_cam[i] += R[i][j] * X[j];
            }
            X_cam[i] += t[i];
        }
        T x = X_cam[0] / X_cam[2];
        T y = X_cam[1] / X_cam[2];

        // Apply distortion
        T r2 = x * x + y * y;
        T distortion = T(1.0) + r2 * (k1 + k2 * r2);
        T x_distorted = x * distortion + T(2.0) * p1 * x * y + p2 * (r2 + T(2.0) * x * x);
        T y_distorted = y * distortion + p1 * (r2 + T(2.0) * y * y) + T(2.0) * p2 * x * y;

        // Compute residuals
        residuals[0] = u - (fx * x_distorted + cx);
        residuals[1] = v - (fy * y_distorted + cy);

        return true;
    }

    double u;
    double v;
};


Eigen::MatrixXd solve_ba_8(
    Eigen::Ref<Eigen::VectorXd> camera,
    Eigen::Ref<Eigen::MatrixXd> poses,
    Eigen::Ref<Eigen::MatrixXd> points,
    const Eigen::VectorXi &poses_i,
    const Eigen::VectorXi &points_i,
    const Eigen::MatrixXd& points_2d,
    bool verbose
) {
    size_t n_pose = poses.rows();
    size_t n_point = points.rows();
    size_t n_obs = points_2d.rows();
    assert(poses_i.size() == n_obs);
    assert(points_i.size() == n_obs);
    // std::cout << n_pose << " " << n_point << " " << n_obs << std::endl;

    ceres::Problem problem;

    // Add intrinsic parameter block
    problem.AddParameterBlock((double*)camera.data(), 8);

    // Add pose parameter blocks
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    std::vector<Vector6d> poses_v(n_pose, Vector6d::Zero());
    for (size_t i = 0; i < n_pose; ++i) {
        poses_v[i] = poses.row(i);
        problem.AddParameterBlock((double*)poses_v[i].data(), 6);
    }

    // Add point parameter blocks
    std::vector<Eigen::Vector3d> points_v(n_point, Eigen::Vector3d::Zero());
    for (size_t i = 0; i < n_point; ++i) {
        points_v[i] = points.row(i);
        problem.AddParameterBlock((double*)points_v.data(), 3);
    }

    // Add reprojection residuals
    std::vector<ceres::ResidualBlockId> residual_blocks;
    for (size_t i = 0; i < n_obs; ++i) {
        int pose_idx = poses_i[i];
        int point_idx = points_i[i];
        assert(pose_idx >= 0 && pose_idx < n_pose);
        assert(point_idx >= 0 && point_idx < n_point);

        const Eigen::Vector2d& uv = points_2d.row(i);

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionCost8, 2, 8, 6, 3>(
                new ReprojectionCost8(uv(0), uv(1)));

        ceres::LossFunction* loss_function =
            new ceres::HuberLoss(2.0);

        ceres::ResidualBlockId residual = problem.AddResidualBlock(
            cost_function, loss_function,
            (double*)camera.data(),
            (double*)poses_v[pose_idx].data(),
            (double*)points_v[point_idx].data());
        
        residual_blocks.push_back(residual);
    }

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = verbose;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-6;
    options.parameter_tolerance = 1e-5;

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

    // get residuals
    Eigen::MatrixXd residuals(n_obs, 2);
    std::vector<double> evaluated_residuals;
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.residual_blocks = residual_blocks;
    problem.Evaluate(eval_options, nullptr, &evaluated_residuals, nullptr, nullptr);
    for (int i = 0; i < n_obs; i++)
        residuals.row(i) << evaluated_residuals[2*i], evaluated_residuals[2*i+1];
    return residuals;
}



// f, cx, cy

struct ReprojectionCost3 {
    ReprojectionCost3(double u, double v)
        : u(u), v(v) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const pose_params,
                    const T* const point_params,
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

        // so3 exponentiation
        T theta = ceres::sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2] + T(1e-12));
        T n[3] = { r[0]/theta, r[1]/theta, r[2]/theta };
        T R[3][3] = {
            { T(0.0), -n[2], n[1] },
            { n[2], T(0.0), -n[0] },
            { -n[1], n[0], T(0.0) }
        };
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R[i][j] = ceres::cos(theta) * T(i==j ? 1.0 : 0.0) +
                    (T(1.0) - ceres::cos(theta)) * n[i] * n[j] +
                    ceres::sin(theta) * R[i][j];
            }
        }

        // Compute projection
        T X_cam[3];
        for (int i = 0; i < 3; ++i) {
            X_cam[i] = T(0);
            for (int j = 0; j < 3; ++j) {
                X_cam[i] += R[i][j] * X[j];
            }
            X_cam[i] += t[i];
        }
        T x = X_cam[0] / X_cam[2];
        T y = X_cam[1] / X_cam[2];

        // Compute residuals
        residuals[0] = u - (f * x + cx);
        residuals[1] = v - (f * y + cy);

        return true;
    }

    double u;
    double v;
};

Eigen::MatrixXd solve_ba_3(
    Eigen::Ref<Eigen::VectorXd> camera,
    Eigen::Ref<Eigen::MatrixXd> poses,
    Eigen::Ref<Eigen::MatrixXd> points,
    const Eigen::VectorXi &poses_i,
    const Eigen::VectorXi &points_i,
    const Eigen::MatrixXd& points_2d,
    bool verbose
) {
    size_t n_pose = poses.rows();
    size_t n_point = points.rows();
    size_t n_obs = points_2d.rows();
    assert(poses_i.size() == n_obs);
    assert(points_i.size() == n_obs);
    // std::cout << n_pose << " " << n_point << " " << n_obs << std::endl;

    ceres::Problem problem;

    // Add intrinsic parameter block
    problem.AddParameterBlock((double*)camera.data(), 3);

    // Add pose parameter blocks
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    std::vector<Vector6d> poses_v(n_pose, Vector6d::Zero());
    for (size_t i = 0; i < n_pose; ++i) {
        poses_v[i] = poses.row(i);
        problem.AddParameterBlock((double*)poses_v[i].data(), 6);
    }

    // Add point parameter blocks
    std::vector<Eigen::Vector3d> points_v(n_point, Eigen::Vector3d::Zero());
    for (size_t i = 0; i < n_point; ++i) {
        points_v[i] = points.row(i);
        problem.AddParameterBlock((double*)points_v.data(), 3);
    }

    // Add reprojection residuals
    std::vector<ceres::ResidualBlockId> residual_blocks;
    for (size_t i = 0; i < n_obs; ++i) {
        int pose_idx = poses_i[i];
        int point_idx = points_i[i];
        assert(pose_idx >= 0 && pose_idx < n_pose);
        assert(point_idx >= 0 && point_idx < n_point);

        const Eigen::Vector2d& uv = points_2d.row(i);

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionCost3, 2, 3, 6, 3>(
                new ReprojectionCost3(uv(0), uv(1)));

        ceres::LossFunction* loss_function =
            new ceres::HuberLoss(2.0);

        ceres::ResidualBlockId residual = problem.AddResidualBlock(
            cost_function, loss_function,
            (double*)camera.data(),
            (double*)poses_v[pose_idx].data(),
            (double*)points_v[point_idx].data());
        
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

    // get residuals
    Eigen::MatrixXd residuals(n_obs, 2);
    std::vector<double> evaluated_residuals;
    ceres::Problem::EvaluateOptions eval_options;
    eval_options.residual_blocks = residual_blocks;
    problem.Evaluate(eval_options, nullptr, &evaluated_residuals, nullptr, nullptr);
    for (int i = 0; i < n_obs; i++)
        residuals.row(i) << evaluated_residuals[2*i], evaluated_residuals[2*i+1];
    return residuals;
}


PYBIND11_MODULE(ba_solver, m) {
    m.def("solve_ba_8", &solve_ba_8, "");
    m.def("solve_ba_3", &solve_ba_3, "");
}
