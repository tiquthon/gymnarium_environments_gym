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

use gymnarium_base::space::{DimensionBoundaries, DimensionValue, SpaceError};
use gymnarium_base::{
    ActionSpace, AgentAction, Environment, EnvironmentState, ObservationSpace, Seed, ToActionMapper,
};

use gymnarium_visualisers_base::input::{Button, ButtonState, Input, Key};
use gymnarium_visualisers_base::{
    Color, DrawableEnvironment, Geometry2D, LineShape, Position2D, Size2D,
    TwoDimensionalDrawableEnvironment, Vector2D, Viewport2D, Viewport2DModification,
};

use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

/* --- --- --- GENERAL --- --- --- */

#[derive(Debug)]
pub enum MountainCarError {
    InternalSpaceError(SpaceError),
    GivenActionDoesNotFitActionSpace,
}

impl std::fmt::Display for MountainCarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalSpaceError(space_error) => {
                write!(f, "An internal space error occurred ({})", space_error)
            }
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

pub struct MountainCar {
    goal_velocity: f64,
    environment_state: EnvironmentState,
    rng: ChaCha20Rng,
}

impl MountainCar {
    pub fn new(goal_velocity: f64) -> Self {
        Self {
            goal_velocity,
            environment_state: Self::observation_space().sample(),
            rng: ChaCha20Rng::from_entropy(),
        }
    }

    fn position(&self) -> f32 {
        if let DimensionValue::FLOAT(f) = self
            .environment_state
            .get_value(&[0])
            .expect("Could not extract position out of environment state.")
        {
            *f
        } else {
            panic!("Position in environment state is no FLOAT.")
        }
    }

    fn velocity(&self) -> f32 {
        if let DimensionValue::FLOAT(f) = self
            .environment_state
            .get_value(&[1])
            .expect("Could not extract velocity out of environment state.")
        {
            *f
        } else {
            panic!("Velocity in environment state is no FLOAT.")
        }
    }
}

impl Environment<MountainCarError, ()> for MountainCar {
    fn action_space() -> ActionSpace {
        ActionSpace::simple(vec![DimensionBoundaries::from(-1..=1)])
    }

    fn observation_space() -> ObservationSpace {
        ObservationSpace::simple(vec![
            DimensionBoundaries::from(MINIMUM_POSITION..=MAXIMUM_POSITION),
            DimensionBoundaries::from(-MAXIMUM_SPEED..=MAXIMUM_SPEED),
        ])
    }

    fn suggested_episode_steps_count() -> Option<u128> {
        Some(200)
    }

    fn reseed(&mut self, random_seed: Option<Seed>) -> Result<(), MountainCarError> {
        if let Some(seed) = random_seed {
            self.rng = ChaCha20Rng::from_seed(seed.into());
        } else {
            self.rng = ChaCha20Rng::from_entropy();
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<EnvironmentState, MountainCarError> {
        self.environment_state = Self::observation_space().sample_with(&mut self.rng);
        Ok(self.environment_state.clone())
    }

    fn step(
        &mut self,
        action: &AgentAction,
    ) -> Result<(EnvironmentState, f64, bool, ()), MountainCarError> {
        if !Self::action_space().contains(action) {
            Err(MountainCarError::GivenActionDoesNotFitActionSpace)
        } else {
            let mut position = self.position();
            let mut velocity = self.velocity();

            let direction = if let DimensionValue::INTEGER(f) = action
                .get_value(&[0])
                .expect("Could not extract direction out of action.")
            {
                *f
            } else {
                panic!("Given direction is no INTEGER.")
            };

            velocity += direction as f32 * FORCE + (3f32 * position).cos() * (-GRAVITY);
            velocity = clamp(velocity, -MAXIMUM_SPEED, MAXIMUM_SPEED);

            position += velocity;
            position = clamp(position, MINIMUM_POSITION, MAXIMUM_POSITION);

            if position == MINIMUM_POSITION && velocity < 0f32 {
                velocity = 0f32;
            }

            let done = position >= GOAL_POSITION && velocity >= self.goal_velocity as f32;
            let reward = -1.0f64;

            self.environment_state = EnvironmentState::simple(vec![
                DimensionValue::FLOAT(position),
                DimensionValue::FLOAT(velocity),
            ]);
            Ok((self.environment_state.clone(), reward, done, ()))
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

        let position = self.position();
        let car = Geometry2D::group(vec![chassis, front_wheel, back_wheel])
            .move_by(Vector2D::with(
                (position - MINIMUM_POSITION) as f64 * scale as f64,
                height_calculator(position as f64) * scale as f64,
            ))
            .rotate_around_self((3f64 * position as f64).cos());

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

#[derive(Default)]
struct MountainCarInputToActionMapper {
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

impl Environment<MountainCarError, ()> for MountainCarContinuous {
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

    fn step(
        &mut self,
        _action: &AgentAction,
    ) -> Result<(EnvironmentState, f64, bool, ()), MountainCarError> {
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
