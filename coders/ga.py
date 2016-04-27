from typing import List
from abc import ABCMeta, abstractmethod, abstractclassmethod
from .util import rint
import datetime
import numpy as np


class Individual(metaclass=ABCMeta):

    @abstractmethod
    def mutate(self, amount: float, scale: float) -> 'Individual':
        pass

    @abstractmethod
    def cross(self, other: 'Individual', amount: float) -> 'Individual':
        pass

    @abstractclassmethod
    def random(cls) -> 'Individual':
        pass


class Population(object):

    def __init__(self, cls: Individual, fn, population_size: int=20,
                 fitness_target: float=0, retain_amount: float=0.2,
                 random_select: float=0.1, mutate_chance: float=0.2,
                 mutate_amount: float=0.3, mutate_scale: float=1,
                 cross_amount: float=0.3):
        self.fn = fn
        self.size = population_size
        self.fitness_target = fitness_target
        self.retain_amount = retain_amount
        self.retain_num = rint(self.size * self.retain_amount)
        self.random_select = random_select
        self.mutate_chance = mutate_chance
        self.mutate_amount = mutate_amount
        self.mutate_scale = mutate_scale
        self.cross_amount = cross_amount
        self.gen = 0
        if isinstance(cls, list):
            self.pop = [cls]
        else:
            self.pop = [[cls.random() for _ in range(self.size)]]
        self.grades = [self._grade()]

    @property
    def current_population(self) -> List[Individual]:
        return self.pop[self.gen]

    @property
    def current_grade(self) -> List[float]:
        return self.grades[self.gen]

    def _grade(self, generation: int=-1) -> List[float]:
        return [self.fn(i) for i in self.pop[generation]]

    def _sort(self, target: float=0, generation: int=-1) -> List[Individual]:
        return [i for g, i in sorted(zip(self.grades[generation],
                                         self.pop[generation]),
                                     key=lambda t: abs(target - t[0]))]

    def set_fitness_fn(self, fn):
        self.fn = fn

    def evolve(self, target: float=0, max_iter: int=20,
               max_time: int=600):
        start = datetime.datetime.utcnow()
        run_end = start
        deadline = start + datetime.timedelta(seconds=max_time)
        runs = []
        for i in range(max_iter):
            # Start timer
            run_start = datetime.datetime.utcnow()

            # Grade and sort
            sorted_ = self._sort(target)

            # Select best parents
            best = sorted_[:self.retain_num]

            # Select lucky parents
            lucky = [i for i in sorted_[self.retain_num:]
                     if self.random_select > np.random.random()]

            # Mutate parents
            parents = [i.mutate(self.mutate_amount, self.mutate_scale)
                       if self.mutate_chance > np.random.random()
                       else i for i in best + lucky]

            # Populate children
            num_parents = len(parents)
            num_children = self.size - num_parents
            children = []
            while len(children) < num_children:
                m = np.random.randint(num_parents)
                f = np.random.randint(num_parents)
                if m != f:
                    children.append(parents[m].cross(parents[f],
                                                     self.cross_amount))

            self.gen += 1
            self.pop.append(parents + children)
            self.grades.append(self._grade())
            top = sorted(self.current_grade, reverse=True)
            top25 = top[:len(self.current_grade) // 4]
            overall_grade = round(sum(top) / len(top))
            top25_grade = round(sum(top25) / len(top25))
            top_grade = round(top[0])
            print('Gen {} grades: {}/{}/{}'.format(self.gen, top_grade,
                                                   top25_grade, overall_grade))

            # End timer
            run_end = datetime.datetime.utcnow()
            runs.append(run_end - run_start)
            run_average = sum(runs[-5:], datetime.timedelta()) / len(runs[-5:])
            print('Gen average time:', run_average)
            if deadline - run_end < run_average:
                print('Timing out after:', run_end - start)
                break
        print('Average run:', sum(runs, datetime.timedelta()) / len(runs))
        print('Done in:', run_end - start)


class Series(Individual):

    def __init__(self):
        self.numbers = [np.random.randint(21) for _ in range(10)]

    def mutate(self, amount: float, scale: float) -> 'Series':
        s = Series()
        s.numbers = [n + int(scale * np.random.randint(-5, 6))
                     if amount > np.random.random()
                     else n for n in self.numbers]
        return s

    def cross(self, other: 'Series', amount: float) -> 'Series':
        s = Series()
        s.numbers = []
        for i, v in enumerate(self.numbers):
            o = other.numbers[i]
            if amount > np.random.random():
                n = (v + o) // 2
            elif np.random.randint(2):
                n = v
            else:
                n = o
            s.numbers.append(n)
        return s

    @classmethod
    def random(cls):
        return Series()

    def __repr__(self):
        return str(self.numbers)


def series_fitness(s: Series):
    return abs(180 - sum(s.numbers)) - 1000
