import unittest
from tsvalidation.data_generation.frequency_modulation import (
    FrequencyModulationWithStep,
    FrequencyModulationLinear,
)


class TestFrequencyModulationWithStep(unittest.TestCase):
    def test_initialization(self):
        with self.assertRaises(TypeError):
            FrequencyModulationWithStep("freq_init", 10)
        with self.assertRaises(TypeError):
            FrequencyModulationWithStep(10, "t_split")
        with self.assertRaises(ValueError):
            FrequencyModulationWithStep(-1, 10)
        with self.assertRaises(ValueError):
            FrequencyModulationWithStep(10, -1)

    def test_modulate(self):
        fm = FrequencyModulationWithStep(10, 5)
        self.assertEqual(fm.modulate(3), 10)
        self.assertEqual(fm.modulate(6), 20)


class TestFrequencyModulationLinear(unittest.TestCase):
    def test_initialization(self):
        with self.assertRaises(TypeError):
            FrequencyModulationLinear("freq_init", 0.5)
        with self.assertRaises(TypeError):
            FrequencyModulationLinear(10, "slope")
        with self.assertRaises(ValueError):
            FrequencyModulationLinear(-1, 0.5)
        with self.assertRaises(ValueError):
            FrequencyModulationLinear(10, -0.5)

    def test_modulate(self):
        fm = FrequencyModulationLinear(10, 0.5)
        self.assertEqual(fm.modulate(3), 10 + 0.5 * 3)


if __name__ == "__main__":
    unittest.main()
