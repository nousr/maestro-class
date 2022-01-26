# gibbon

Implementations of adversarial attack & defense methods in ml.

---

## Best Performance

- PGD
    - Params:
        ```
        eps: 8.0/255.0,
        eps_iter: 2.0/255.0,
        steps: 7
        ```
    - Results:
        ```
        Test Accuracy: 82/90
        Target Label: 7
        Attack Success Rate: 91%
        Distance: 1.3890866
        Number of Queries: 720
        Leaderboard Score: ~91.12566
        ```

## Resources

- [GAMA-FW](https://arxiv.org/abs/2011.14969):
    - Relatively new method (2020).
    - Cited as being a strong attacker when the number of steps is restricted. 
    - Has documented success on the CIFAR dataset.
    - Has source code for reference.
- [Towards Evaluating the Robustness...](https://arxiv.org/abs/1608.04644):
    - Outlines attacks tailored to three distance metrics.
    - Shows that defnse distillation does not defend against the attacks.
    - Minimum distance
