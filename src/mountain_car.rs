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

use gymnarium_base::{Environment, ObservationSpace, ActionSpace, Seed, EnvironmentState, AgentAction};
use gymnarium_base::space::{DimensionBoundaries, SpaceError, DimensionValue};

use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::SeedableRng;

/* --- --- --- GENERAL --- --- --- */

#[derive(Debug)]
pub enum MountainCarError {
    InternalSpaceError(SpaceError),
    GivenActionDoesNotFitActionSpace
}

impl std::fmt::Display for MountainCarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InternalSpaceError(space_error) =>
                write!(f, "An internal space error occurred ({})", space_error),
            Self::GivenActionDoesNotFitActionSpace =>
                write!(f, "Given Action does not fit ActionSpace"),
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
}

impl Environment<MountainCarError, ()> for MountainCar {
    fn action_space() -> ActionSpace {
        ActionSpace::simple(vec![
            DimensionBoundaries::from(-1..=1)
        ])
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

    fn step(&mut self, action: &AgentAction) -> Result<(EnvironmentState, f64, bool, ()), MountainCarError> {
        if !Self::action_space().contains(action) {
            Err(MountainCarError::GivenActionDoesNotFitActionSpace)
        } else {
            let mut position = if let DimensionValue::FLOAT(f) = self.environment_state
                .get_value(&[0])
                .map_err(MountainCarError::InternalSpaceError)? { *f } else { panic!() };

            let mut velocity =if let DimensionValue::FLOAT(f) = self.environment_state
                .get_value(&[1])
                .map_err(MountainCarError::InternalSpaceError)? { *f } else { panic!() };

            let direction = if let DimensionValue::INTEGER(f) = action
                .get_value(&[0])
                .map_err(MountainCarError::InternalSpaceError)? { *f } else { panic!() };

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
                DimensionValue::FLOAT(position), DimensionValue::FLOAT(velocity)
            ]);
            Ok((self.environment_state.clone(), reward, done, ()))
        }
    }

    fn close(&mut self) -> Result<(), MountainCarError> {
        Ok(())
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

    fn step(&mut self, _action: &AgentAction) -> Result<(EnvironmentState, f64, bool, ()), MountainCarError> {
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
