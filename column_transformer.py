def ColumnTransformer(df):    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import make_column_selector, make_column_transformer
    import numpy as np
    import pandas as pd

    # pulls column names from number columns only
    num_attribs = df.select_dtypes(include=[np.number]).columns 

    # pulls columns names from object names only
    cat_attribs = df.select_dtypes(include=[object]).columns 

    num_pipeline = make_pipeline(
                                SimpleImputer(strategy="median"), 
                                StandardScaler())
    
    cat_pipeline = make_pipeline(
                                SimpleImputer(strategy="most_frequent"),
                                OneHotEncoder(handle_unknown="ignore"))

    # make a quicker preprocessing pipeline
    preprocessing = make_column_transformer(
        (num_pipeline, make_column_selector(dtype_include=np.number)),
        (cat_pipeline, make_column_selector(dtype_include=object)))
        
    return preprocessing