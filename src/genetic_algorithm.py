import string
import random
from typing import List


class GeneticAlgorithm:
    POSSIBLE_CHARACTERS = string.ascii_uppercase + '_' + string.digits

    def __init__(self, target_string: str, possible_characters: List[str] = POSSIBLE_CHARACTERS):
        self.target_string = target_string
        self.possible_characters = possible_characters
        self.population_size = 1000
        self.mutation_rate = 1 / len(self.target_string)
        self.generations = 1000
        self.selected_population_size = int(0.2 * self.population_size)

    def generate_population(self) -> List[str]:
        return [''.join(random.choices(self.POSSIBLE_CHARACTERS, k=len(self.target_string)))
                for _ in range(self.population_size)]

    def fitness(self, child: str) -> int:
        return sum([i == j for i, j in zip(child, self.target_string)])

    @staticmethod
    def crossover(p1: str, p2: str, method='uniform') -> str:
        if method == 'uniform':
            return ''.join(random.choice([i, j]) for i, j in zip(p1, p2))
        else:
            raise NotImplementedError("only uniform method is implemented")

    def mutate(self, original: str) -> str:
        flag_mutate = [random.random() < self.mutation_rate for _ in range(len(original))]
        what_to_mutate = random.choices(self.possible_characters, k=len(original))
        return ''.join(what_to if flag else ori
                       for ori, what_to, flag in zip(original, what_to_mutate, flag_mutate))

    def evolve_population(self, population: List[str]) -> List[str]:
        scores = [self.fitness(child) for child in population]
        selected = sorted(population, key=self.fitness, reverse=True)[:self.selected_population_size]
        next_generation = []
        for _ in range(self.population_size):
            p1, p2 = random.choices(selected, weights=[self.fitness(c) for c in selected], k=2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            next_generation.append(child)
        return next_generation

    def run(self):
        population = self.generate_population()
        for generation in range(self.generations):
            print(f"Generation: {generation}, Best Score: {self.fitness(population[0])}, String: {population[0]}")
            if population[0] == self.target_string:
                break
            population = self.evolve_population(population)


if __name__ == "__main__":
    GeneticAlgorithm("VICTORIA_3_PARADOX").run()
