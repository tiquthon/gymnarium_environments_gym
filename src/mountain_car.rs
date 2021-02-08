//! # MountainCar
//!
//! Currently there are two versions of MountainCar.
//! One with a discrete action space and one with a continuous one.
//!
//! ## Discrete version
//!
//! The description on the OpenAI page reads:
//!
//! > *&ldquo;A car is on a one-dimensional track, positioned between two "mountains". The goal
//! > is to drive up the mountain on the right; however, the car's engine is not strong enough to
//! > scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and
//! > forth to build up momentum.&rdquo;*
//!
//! This discrete version accepts one of three actions: Left, Right or Do Nothing.
//!
//! ## Continuous version
//!
//! There is also the continuous version.
//! In this version the action space accepts a value between -1.0 and 1.0 and the control can thus
//! be more precise.
//!
//! The description on the OpenAI page has been appended with:
//!
//! > *&ldquo;... Here, the reward is greater if you spend less energy to reach the goal&rdquo;*
//!
//! ---
//!
//! *These environments are taken from
//! [OpenAI Gym MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/) and
//! [OpenAI Gym MountainCarContinuous-v0](https://gym.openai.com/envs/MountainCarContinuous-v0/).*

use gymnarium_base::math::{radians_to_degrees, Position2D, Size2D, Vector2D};
use gymnarium_base::serde::{Deserialize, Serialize};
use gymnarium_base::space::{DimensionBoundaries, DimensionValue};
use gymnarium_base::{
    ActionSpace, AgentAction, Environment, EnvironmentState, ObservationSpace, Seed, ToActionMapper,
};

use gymnarium_visualisers_base::input::{Button, ButtonState, Input, Key};
use gymnarium_visualisers_base::{
    Color, DrawableEnvironment, Geometry2D, LineShape, TwoDimensionalDrawableEnvironment,
    Viewport2D, Viewport2DModification,
};

use rand::distributions::{Distribution, Uniform};

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

/* --- --- --- GENERAL --- --- --- */

#[derive(Debug)]
pub enum MountainCarError {
    GivenActionDoesNotFitActionSpace,
}

impl std::fmt::Display for MountainCarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GivenActionDoesNotFitActionSpace => {
                write!(f, "Given Action does not fit ActionSpace")
            }
        }
    }
}

impl std::error::Error for MountainCarError {}

const MINIMUM_POSITION: f32 = -1.2f32;
const MAXIMUM_POSITION: f32 = 0.6f32;
const MAXIMUM_SPEED: f32 = 0.07f32;
const GRAVITY: f32 = 0.0025f32;
const FORCE: f32 = 0.001f32;
const GOAL_POSITION: f32 = 0.5f32;

/* --- --- --- DISCRETE MOUNTAIN CAR --- --- --- */

/// The goal is to drive up the mountain on the right.
///
/// The agent (a car) is started at the bottom of a valley. For any given state the agent may
/// choose to accelerate to the left, right or cease any acceleration.
///
/// *(Code semantic copied from <https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py>.)*
///
/// ## Source
/// The environment appeared first in Andrew Moore's PhD Thesis (1990).
///
/// ## Observation
/// Space-Structure: `[2]`
///
/// | Index | Observation | Min | Max |
/// | --- | --- | --- | --- |
/// | `[0]` | Car Position | `-1.2` | `0.6` |
/// | `[1]` | Car Velocity | `-0.07` | `0.07` |
///
/// ## Actions
/// Space-Structure: `[1]`
///
/// | Value | Action |
/// | --- | --- |
/// | `[-1]` | Accelerate to the Left |
/// | `[0]` | Don't accelerate |
/// | `[1]` | Accelerate to the Right |
///
/// ## Reward
/// Reward of `0` is awarded if the agent reached the flag (position = `0.5`) on top of the mountain.
///
/// Reward of `-1` is awarded if the position of the agent is less than `0.5`.
///
/// ## Starting State
/// The position of the car is assigned a uniform random value in `[-0.6, -0.4]`.
///
/// The starting velocity of the car is always assigned to `0`.
///
/// ## Episode Termination
/// - The car position is more than `0.5`
/// - Episode length is greater than `200`
///
pub struct MountainCar {
    goal_velocity: f64,
    position: f32,
    velocity: f32,
    last_seed: Seed,
    rng: ChaCha20Rng,
}

impl MountainCar {
    pub fn new(goal_velocity: f64) -> Self {
        let last_seed = Seed::new_random();
        Self {
            goal_velocity,
            position: -0.5f32,
            velocity: 0f32,
            last_seed: last_seed.clone(),
            rng: ChaCha20Rng::from_seed(last_seed.into()),
        }
    }
}

impl Environment<MountainCarError, (), MountainCarStorage> for MountainCar {
    fn action_space() -> ActionSpace {
        ActionSpace::simple(vec![DimensionBoundaries::INTEGER(-1, 1)])
    }

    fn observation_space() -> ObservationSpace {
        ObservationSpace::simple(vec![
            DimensionBoundaries::FLOAT(MINIMUM_POSITION, MAXIMUM_POSITION),
            DimensionBoundaries::FLOAT(-MAXIMUM_SPEED, MAXIMUM_SPEED),
        ])
    }

    fn suggested_episode_steps_count() -> Option<u128> {
        Some(200)
    }

    fn reseed(&mut self, random_seed: Option<Seed>) -> Result<(), MountainCarError> {
        if let Some(seed) = random_seed {
            self.last_seed = seed;
            self.rng = ChaCha20Rng::from_seed(self.last_seed.clone().into());
        } else {
            self.last_seed = Seed::new_random();
            self.rng = ChaCha20Rng::from_seed(self.last_seed.clone().into());
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<EnvironmentState, MountainCarError> {
        self.position = Uniform::new_inclusive(-0.6f32, -0.4f32).sample(&mut self.rng);
        self.velocity = 0f32;
        Ok(self.state())
    }

    fn state(&self) -> EnvironmentState {
        EnvironmentState::simple(vec![
            DimensionValue::FLOAT(self.position),
            DimensionValue::FLOAT(self.velocity),
        ])
    }

    fn step(
        &mut self,
        action: &AgentAction,
    ) -> Result<(EnvironmentState, f64, bool, ()), MountainCarError> {
        if !Self::action_space().contains(action) {
            Err(MountainCarError::GivenActionDoesNotFitActionSpace)
        } else {
            let direction = action[&[0]].expect_integer();

            self.velocity += direction as f32 * FORCE + (3f32 * self.position).cos() * (-GRAVITY);
            self.velocity = clamp(self.velocity, -MAXIMUM_SPEED, MAXIMUM_SPEED);

            self.position += self.velocity;
            self.position = clamp(self.position, MINIMUM_POSITION, MAXIMUM_POSITION);

            if self.position == MINIMUM_POSITION && self.velocity < 0f32 {
                self.velocity = 0f32;
            }

            let done = self.position >= GOAL_POSITION && self.velocity >= self.goal_velocity as f32;
            let reward = if done { 0f64 } else { -1.0f64 };

            Ok((self.state(), reward, done, ()))
        }
    }

    fn load(&mut self, data: MountainCarStorage) -> Result<(), MountainCarError> {
        self.goal_velocity = data.goal_velocity;
        self.position = data.position;
        self.velocity = data.velocity;
        self.last_seed = data.last_seed.clone();
        self.rng = ChaCha20Rng::from_seed(self.last_seed.clone().into());
        self.rng.set_word_pos(data.rng_word_pos);
        Ok(())
    }

    fn store(&self) -> MountainCarStorage {
        MountainCarStorage {
            goal_velocity: self.goal_velocity,
            position: self.position,
            velocity: self.velocity,
            last_seed: self.last_seed.clone(),
            rng_word_pos: self.rng.get_word_pos(),
        }
    }

    fn close(&mut self) -> Result<(), MountainCarError> {
        Ok(())
    }
}

impl DrawableEnvironment for MountainCar {
    fn suggested_rendered_steps_per_second() -> Option<f64> {
        Some(60f64)
    }
}

impl TwoDimensionalDrawableEnvironment<MountainCarError> for MountainCar {
    fn draw_two_dimensional(&self) -> Result<Vec<Geometry2D>, MountainCarError> {
        let height_calculator: fn(f64) -> f64 = |x| (3f64 * x).sin() * 0.45f64 + 0.55f64;

        // render

        let screen_width = 600f32;

        let world_width = MAXIMUM_POSITION - MINIMUM_POSITION;
        let scale = screen_width / world_width;
        let carwidth = 40f64;
        let carheight = 20f64;

        // track
        let track = Geometry2D::polyline(
            (0..=100)
                .into_iter()
                .map(|index| {
                    index as f32 / 100f32 * (MAXIMUM_POSITION - MINIMUM_POSITION) + MINIMUM_POSITION
                })
                .map(|x| x as f64)
                .map(|x| {
                    Position2D::with(
                        (x - MINIMUM_POSITION as f64) * scale as f64,
                        height_calculator(x) * scale as f64,
                    )
                })
                .collect(),
        )
        .line_shape(LineShape::Round)
        .line_or_border_color(Color::black())
        .line_or_border_width(2f64);

        // car
        let clearance = 10f64;

        let (l, r, t, b) = (-carwidth / 2f64, carwidth / 2f64, carheight, 0f64);
        let chassis = Geometry2D::polygon(vec![
            Position2D::with(l, b),
            Position2D::with(l, t),
            Position2D::with(r, t),
            Position2D::with(r, b),
        ])
        .move_by(Vector2D::with(0f64, clearance));

        let gray = (255f64 * 0.5f64) as u8;
        let front_wheel = Geometry2D::circle(Position2D::zero(), carheight / 2.5f64)
            .fill_color(Color::with(gray, gray, gray, 255))
            .move_by(Vector2D::with(carwidth / 4f64, clearance));

        let back_wheel = Geometry2D::circle(Position2D::zero(), carheight / 2.5f64)
            .fill_color(Color::with(127, 127, 127, 255))
            .move_by(Vector2D::with(-carwidth / 4f64, clearance));

        let car = Geometry2D::group(vec![chassis, front_wheel, back_wheel])
            .move_by(Vector2D::with(
                (self.position - MINIMUM_POSITION) as f64 * scale as f64,
                height_calculator(self.position as f64) * scale as f64,
            ))
            .rotate_around_self(radians_to_degrees((3f64 * self.position as f64).cos()));

        // flag
        let flagx = (GOAL_POSITION - MINIMUM_POSITION) as f64 * scale as f64;
        let flagy1 = height_calculator(GOAL_POSITION as f64) * scale as f64;
        let flagy2 = flagy1 + 50f64;
        let flagpole = Geometry2D::line(
            Position2D::with(flagx, flagy1),
            Position2D::with(flagx, flagy2),
        )
        .line_or_border_color(Color::black())
        .line_or_border_width(1f64);

        let flag_color = (255f64 * 0.8f64) as u8;
        let flag = Geometry2D::polygon(vec![
            Position2D::with(flagx, flagy2),
            Position2D::with(flagx, flagy2 - 10f64),
            Position2D::with(flagx + 25f64, flagy2 - 5f64),
        ])
        .fill_color(Color::with(flag_color, flag_color, 0, 255));

        Ok(vec![track, car, flagpole, flag])
    }

    fn preferred_view(&self) -> Option<(Viewport2D, Viewport2DModification)> {
        Some((
            Viewport2D::with(
                Position2D::with(300f64, 200f64),
                Size2D::with(600f64, 400f64),
            ),
            Viewport2DModification::KeepAspectRatioAndScissorRemains,
        ))
    }

    fn preferred_background_color(&self) -> Option<Color> {
        Some(Color::white())
    }
}

#[derive(Serialize, Deserialize)]
pub struct MountainCarStorage {
    goal_velocity: f64,
    position: f32,
    velocity: f32,
    last_seed: Seed,
    rng_word_pos: u128,
}

#[derive(Default)]
pub struct MountainCarInputToActionMapper {
    left_pressed: bool,
    right_pressed: bool,
}

impl ToActionMapper<Vec<Input>, MountainCarError> for MountainCarInputToActionMapper {
    fn map(&mut self, inputs: &Vec<Input>) -> Result<AgentAction, MountainCarError> {
        for input in inputs {
            if let Input::Button(button_args) = input {
                if let Button::Keyboard(key) = button_args.button {
                    match key {
                        Key::Left => {
                            self.left_pressed = button_args.state == ButtonState::Press;
                        }
                        Key::Right => {
                            self.right_pressed = button_args.state == ButtonState::Press;
                        }
                        _ => (),
                    }
                }
            }
        }
        Ok(AgentAction::simple(vec![DimensionValue::from(
            if self.left_pressed && !self.right_pressed {
                -1
            } else if self.right_pressed && !self.left_pressed {
                1
            } else {
                0
            },
        )]))
    }
}

/* --- --- --- CONTINUOUS MOUNTAIN CAR --- --- --- */

pub struct MountainCarContinuous;

impl Environment<MountainCarError, (), MountainCarStorage> for MountainCarContinuous {
    fn action_space() -> ActionSpace {
        todo!()
    }

    fn observation_space() -> ObservationSpace {
        todo!()
    }

    fn suggested_episode_steps_count() -> Option<u128> {
        todo!()
    }

    fn reseed(&mut self, _random_seed: Option<Seed>) -> Result<(), MountainCarError> {
        todo!()
    }

    fn reset(&mut self) -> Result<EnvironmentState, MountainCarError> {
        todo!()
    }

    fn state(&self) -> EnvironmentState {
        todo!()
    }

    fn step(
        &mut self,
        _action: &AgentAction,
    ) -> Result<(EnvironmentState, f64, bool, ()), MountainCarError> {
        todo!()
    }

    fn load(&mut self, _data: MountainCarStorage) -> Result<(), MountainCarError> {
        todo!()
    }

    fn store(&self) -> MountainCarStorage {
        todo!()
    }

    fn close(&mut self) -> Result<(), MountainCarError> {
        todo!()
    }
}

/* --- --- --- HELPER FUNCTIONS --- --- --- */

fn min<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a < b {
        a
    } else {
        b
    }
}

fn max<T: PartialOrd + Copy>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

fn clamp<T: PartialOrd + Copy>(value: T, minimum: T, maximum: T) -> T {
    min(max(value, minimum), maximum)
}
