import xgboost as xgb

MODEL_PATH = "ml_models/mbd-opt-sapt.json" #corresponds to the MBD-S-ML​ dispersion model in the paper

def main():
    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")

    top20 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

    print("Top 20 features for MBD-S-ML:\n")
    for name, gain in top20:
        print(f"{name:45s}  gain = {gain:.6f}")

if __name__ == "__main__":
    main()
