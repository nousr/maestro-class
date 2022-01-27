# gibbon

Implementations of adversarial attack & defense methods in ml.

---

## Best Performance

- PGD
    - Params:
        ```
        eps: 15/255
        eps_iter: 7.5.0/255.0
        steps: 15 (with random restarts & 5 attempts)
        ```
    - Results:
        ```
        Test Accuracy: 90/90
        Target Label: 7
        Attack Success Rate: 100%
        Distance: 1.395
        Number of Queries: 858
        Leaderboard Score: ~97.186
        Leaderboard Score v2: 94.1579
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
