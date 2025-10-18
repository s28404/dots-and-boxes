import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import argparse


class FuzzyCartPoleAgent:
    def __init__(self):
        # Definicja wejść fuzzy
        self.position = ctrl.Antecedent(
            np.arange(-3, 3.1, 0.1), "position"
        )  # pozycja wózka
        self.angle = ctrl.Antecedent(np.arange(-30, 31, 1), "angle")  # kąt kija
        self.angular_velocity = ctrl.Antecedent(
            np.arange(-10, 11, 1), "angular_velocity"
        )  # prędkość kija
        # Wyjście fuzzy
        self.action = ctrl.Consequent(
            np.arange(0, 2, 1), "action"
        )  # 0 = left, 1 = right

        # Funkcje przynależności dla pozycji
        self.position["left"] = fuzz.trimf(self.position.universe, [-3, -3, -0.5])
        self.position["center"] = fuzz.trimf(self.position.universe, [-1, 0, 1])
        self.position["right"] = fuzz.trimf(self.position.universe, [0.5, 3, 3])

        # Funkcje przynależności dla kąta (ujemny = pada w lewo, dodatni = pada w prawo)
        self.angle["negative"] = fuzz.trimf(self.angle.universe, [-30, -30, -2])
        self.angle["zero"] = fuzz.trimf(self.angle.universe, [-5, 0, 5])
        self.angle["positive"] = fuzz.trimf(self.angle.universe, [2, 30, 30])

        # Funkcje przynależności dla prędkości kija (ujemna = pada w lewo, dodatnia = pada w prawo)
        self.angular_velocity["negative"] = fuzz.trimf(
            self.angular_velocity.universe, [-10, -10, -1]
        )
        self.angular_velocity["zero"] = fuzz.trimf(
            self.angular_velocity.universe, [-2, 0, 2]
        )
        self.angular_velocity["positive"] = fuzz.trimf(
            self.angular_velocity.universe, [1, 10, 10]
        )

        # Funkcje przynależności dla akcji
        self.action["left"] = fuzz.trimf(self.action.universe, [0, 0, 1])
        self.action["right"] = fuzz.trimf(self.action.universe, [0, 1, 1])

        # Reguły - priorytet: najpierw utrzymaj kij, potem wracaj do centrum
        # Priorytetowe: balansowanie kija
        rule1 = ctrl.Rule(
            self.angle["negative"] | self.angular_velocity["negative"],
            self.action["left"],
        )
        rule2 = ctrl.Rule(
            self.angle["positive"] | self.angular_velocity["positive"],
            self.action["right"],
        )
        # Pomocnicze: jeśli kij stabilny, wróć do centrum
        rule3 = ctrl.Rule(
            self.position["left"] & self.angle["zero"] & self.angular_velocity["zero"],
            self.action["right"],
        )
        rule4 = ctrl.Rule(
            self.position["right"] & self.angle["zero"] & self.angular_velocity["zero"],
            self.action["left"],
        )
        rule5 = ctrl.Rule(
            self.position["center"]
            & self.angle["zero"]
            & self.angular_velocity["zero"],
            self.action["left"],
        )  # fallback

        # System sterowania
        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.simulator = ctrl.ControlSystemSimulation(self.control_system)

    def act(self, observation):
        # observation = [position, velocity, angle, angular_velocity]
        position = observation[0]
        angle_deg = np.rad2deg(observation[2])
        angular_velocity_deg = np.rad2deg(observation[3])

        # Clamp values to the defined universe ranges
        position = np.clip(position, -3, 3)
        angle_deg = np.clip(angle_deg, -30, 30)
        angular_velocity_deg = np.clip(angular_velocity_deg, -10, 10)

        self.simulator.input["position"] = position
        self.simulator.input["angle"] = angle_deg
        self.simulator.input["angular_velocity"] = angular_velocity_deg

        try:
            self.simulator.compute()
            action = int(round(self.simulator.output["action"]))
        except KeyError:
            # If computation fails, default to moving left (action 0)
            action = 0

        return action


def main():
    parser = argparse.ArgumentParser(description="Fuzzy CartPole Agent")
    parser.add_argument(
        "--render_mode",
        type=str,
        choices=["human", "rgb_array"],
        default="rgb_array",
        help="Tryb renderowania: 'human' dla podglądu na żywo, 'rgb_array' dla nagrywania wideo",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Liczba epizodów do uruchomienia"
    )
    args = parser.parse_args()

    # Tworzenie środowiska z wybranym trybem renderowania
    env = gym.make("CartPole-v1", render_mode=args.render_mode)

    # Dodaj nagrywanie tylko dla trybu rgb_array
    if args.render_mode == "rgb_array":
        env = RecordVideo(
            env,
            video_folder="./videos",
            episode_trigger=lambda ep: True,  # Nagrywaj każdy epizod
            name_prefix="fuzzy-cartpole",
        )

    agent = FuzzyCartPoleAgent()

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        while not (done or truncated):
            action = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep + 1}: Total reward = {total_reward}")

    env.close()
    if args.render_mode == "rgb_array":
        print("Filmiki zapisane w folderze ./videos")


if __name__ == "__main__":
    main()
