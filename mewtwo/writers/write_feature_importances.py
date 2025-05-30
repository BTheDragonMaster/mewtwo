from typing import Optional


def write_feature_importances(feature_to_importance: dict[str, float], out_file: str,
                              feature_to_stdev: Optional[dict[str, float]] = None,
                              sort_by_importance: bool = True) -> None:
    features_and_importances = list(feature_to_importance.items())

    if sort_by_importance:
        features_and_importances.sort(key=lambda x: x[1], reverse=True)

    with open(out_file, 'w') as out:
        if feature_to_stdev:
            out.write("feature_name\tfeature_importance\tstdev\n")
        else:
            out.write("feature_name\tfeature_importance\n")

        for feature, importance in features_and_importances:
            if feature_to_stdev:
                stdev = feature_to_stdev[feature]
                out.write(f"{feature}\t{importance}\t{stdev}\n")
            else:

                out.write(f"{feature}\t{importance}\n")

