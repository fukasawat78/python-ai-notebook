import lightgbm as lgm

def get_model():
    model = lgm.LGBMClassifier(
        max_depth=4,
        random_state=1234
    )
    return model