EXP_CONFIG = {
    "exact": {
        "tabular": [
            {"beta": 0.001, "iters": 100000, "lr": 0.0002,},
            {"beta": 0.01, "iters": 5000, "lr": 0.02,},
            {"beta": 0.1, "iters": 1000, "lr": 0.1,},
        ],
        "neural": [
            {"beta": 0.01, "iters": 10000, "lr": 0.003,},
        ],
    },
    "empirical": {
        "tabular": [
            {"beta": 0.01, "iters": 100000, "lr": 0.0002,},
        ],
        "neural": [
            {"beta": 0.1, "iters": 20000, "lr": 0.001,},
        ],
    },
}
