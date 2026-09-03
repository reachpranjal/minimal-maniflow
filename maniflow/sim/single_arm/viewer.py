import signal
import time

import mujoco
import mujoco.viewer

from .env import SingleArmEnv

_control_dt = 0.02


class LiveViewer:
    def __init__(
        self,
        env: SingleArmEnv,
        *,
        speed: float = 1.0,
        rotate: bool = False,
        rotate_speed: float = 36.0,
    ):
        self.env = env
        self.speed = speed
        self.rotate = rotate
        self.rotate_speed = rotate_speed
        self._viewer = None
        self._start_t = 0.0
        self._stop_requested = False
        self._prev_sigint_handler = None

    def _handle_sigint(self, signum, frame) -> None:
        print("\nStopping.")
        self._stop_requested = True

    def __enter__(self) -> "LiveViewer":
        self._prev_sigint_handler = signal.signal(signal.SIGINT, self._handle_sigint)

        self._viewer = mujoco.viewer.launch_passive(
            self.env.model, self.env.data, show_left_ui=False, show_right_ui=False
        ).__enter__()

        if self.rotate:
            cam = self._viewer.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 2.5
            cam.elevation = -20.0
            cam.lookat[:] = [0.3, 0.0, 0.4]
        self._start_t = time.time()
        return self

    def __exit__(self, *exc) -> None:
        if self._prev_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._prev_sigint_handler)
            self._prev_sigint_handler = None
        if self._viewer is not None:
            self._viewer.__exit__(*exc)
            self._viewer = None

    def is_running(self) -> bool:
        return (
            not self._stop_requested
            and self._viewer is not None
            and self._viewer.is_running()
        )

    def sync(self) -> None:
        if self.rotate:
            self._viewer.cam.azimuth = (
                (time.time() - self._start_t) * self.rotate_speed % 360.0
            )
        self._viewer.sync()

    def pause(self, seconds: float) -> None:
        end = time.time() + seconds / self.speed
        while time.time() < end and self.is_running():
            self.sync()
            time.sleep(_control_dt)

    def step_hook(self, _env: SingleArmEnv) -> bool:
        self.sync()
        time.sleep(_control_dt / self.speed)
        return self.is_running()
