import numpy as np
from types import SimpleNamespace
from opencr_mujoco.evaluation.trajectory_evaluator import TrajectoryEvaluator


def test_tip_release_sampling(tmp_path):
    xml = """
    <mujoco>
        <option gravity="0 0 0" integrator="Euler"/>
        <worldbody>
            <body name="mocap_base"/>
            <body name="link_end"/>
            <body name="EE_pos">
                <joint type="slide" axis="1 0 0"/>
                <geom type="sphere" size="0.02" mass="1"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model_path = tmp_path / "test.xml"
    model_path.write_text(xml)

    hz = 128
    dt = 1 / hz
    hold = 0.125
    n = int(hold * hz)
    times = np.arange(17) * dt

    x_release = dt**2 * n * (n + 1) / 2
    v_release = n * dt

    reference = np.zeros((len(times), 6))
    reference[:, 0] = x_release + v_release * times

    evaluator = TrajectoryEvaluator(
        reference_data_dir=tmp_path,
        results_dir=tmp_path / "results",
    )

    evaluator.reference_loader = SimpleNamespace(
        load_tip_release_data=lambda _: (
            np.array([1., 0., 0., 0., 0., 0.]),
            reference,
            dt,
            times,
            None,
            np.zeros(3),
        )
    )

    result = evaluator.evaluate_tip_release(
        model_path=str(model_path),
        num_links=1,
        test_type="SyntheticTest",
        sim_hz=hz,
        force_ramp_time=0,
        hold_time=hold,
        show_progress=False,
    )

    np.testing.assert_allclose(
        result["sim_tip_poses"][:, 0],
        reference[:, 0],
        atol=1e-12,
        rtol=0,
    )
