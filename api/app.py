import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Implement a /predict endpoint
@app.get("/predict")
def predict(original_title,
            title,
            release_date,
            duration_min,
            description,
            budget,
            original_language,
            status,
            number_of_awards_won,
            number_of_nominations,
            has_collection,
            all_genres,
            top_countries,
            number_of_top_productions,
            available_in_english):
    
    X_pred = pd.DataFrame(data = [[original_title,
                                    title,
                                    release_date,
                                    duration_min,
                                    description,
                                    budget,
                                    original_language,
                                    status,
                                    number_of_awards_won,
                                    number_of_nominations,
                                    has_collection,
                                    all_genres,
                                    top_countries,
                                    number_of_top_productions,
                                    available_in_english
                                    ]],
                        columns=['original_title',
                                'title',
                                'release_date',
                                'duration_min',
                                'description',
                                'budget',
                                'original_language',
                                'status',
                                'number_of_awards_won',
                                'number_of_nominations',
                                'has_collection',
                                'all_genres',
                                'top_countries',
                                'number_of_top_productions',
                                'available_in_english'
                                ])
        
    # load a model model.joblib trained locally
    pipeline = joblib.load('model.joblib')

    # need a predict function
    result = pipeline.predict(X_pred)
    
    # return type: dict
    return {
            "title": X_pred.title[0],
            "popularity": result[0]
           }
