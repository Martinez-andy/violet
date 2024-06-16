import pickle as pkl
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Used to convert state names into one-hot values
supported_states = {
    "connecticut" : np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="float"),
    "delaware" : np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="float"),
    "maine" : np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="float"),
    "massachusetts" : np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="float"),
    "new hampsire" : np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="float"),
    "new jersey" : np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype="float"),
    "new york" : np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="float"),
    "pennsylvania" : np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype="float"),
    "puerto rico" : np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype="float"),
    "rhode island" : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype="float"),
    "vermont" : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype="float"),
    "virgin islands" : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype="float"),
    "west virginia" : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype="float"),
    "wyoming" : np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype="float")
}


state_to_median_value = {
    'alabama': 267100,
    'alaska': 350000,
    'arizona': 435300,
    'arkansas': 246000,
    'california': 793600,
    'colorado': 586100,
    'connecticut': 424900,
    'delaware': 375000,
    'district of Columbia': 640000,
    'florida': 405000,
    'georgia': 389500,
    'hawaii': 714100,
    'idaho': 539000,
    'illinois': 266800,
    'indiana': 242500,
    'iowa': 289900,
    'kansas': 263700,
    'kentucky': 246700,
    'louisiana': 243300,
    'maine': 370000,
    'maryland': 395000,
    'massachusetts': 595700,
    'michigan': 238800,
    'minnesota': 330500,
    'mississippi': 232800,
    'missouri': 243500,
    'montana': 609900,
    'nebraska': 280400,
    'nevada': 479299,
    'new hampshire': 451400,
    'new jersey': 485900,
    'new mexico': 358600,
    'new york': 649000,
    'north carolina': 362200,
    'north dakota': 334075,
    'ohio': 228000,
    'oklahoma': 233900,
    'oregon': 490200,
    'pennsylvania': 268100,
    'rhode island': 455500,
    'south carolina': 360800,
    'south dakota': 300200,
    'tennessee': 418900,
    'texas': 336400,
    'utah': 548900,
    'vermont': 395800,
    'virginia': 415600,
    'washington': 605400,
    'west virginia': 284000,
    'wisconsin': 329000,
    'wyoming': 317000
}

def predict():
    # Load the One-Hot KNN Model
    with open("models/knn.pkl", "rb") as f:
        oneKnnModel = pkl.load(f)
        
    # Load the One-Hot Poly model
    with open("models/poly.pkl", "rb") as f:
        onePolyModel = pkl.load(f)
        
    # Load the median KNN model
    with open("models/medianKNN.pkl", "rb") as f:
        medianKnnModel = pkl.load(f)
    
    # Load the median Poly model
    with open("models/medianPoly.pkl", "rb") as f:
        medianPolyModel = pkl.load(f)
    
    
    # Take in details about the property as input for the models
    num_beds = float(input("How many bedrooms: "))
    num_baths = float(input("How many bathrooms: "))
    acre_lot = float(input("Size of the lot in terms of acres: "))
    sq_ft = float(input("Square footage of the property: "))
    state = input("What state is it in: ").lower()
    
    
    """ Only 14 states in data set, so if state is not a supported one,
    then switch state into Maine. According to 
    https://www.bankrate.com/real-estate/median-home-price/#median-price-by-state
    North Carolina has the most median housing values in the US but NC is not in our
    data set. So we must use a the state that is in our data set but is the closest
    to the median housing value. In this case, Maine seems to be that state. So set
    any unsupported states into maine for compatability
    
    Models don't support missing entries so can't just make this value 0
    """
    if state not in supported_states:
        oneState = "maine"
    else:
        oneState = state
        

    # Take inputs and prep for input into median models
    medianData = np.array([num_beds, num_baths, acre_lot, sq_ft, state_to_median_value[state]]).reshape(1, -1)
    
    
    # Take inputs and prep for input into one-hot models
    state = supported_states[oneState]
    oneData = np.append(np.array([num_beds, num_baths, acre_lot, sq_ft]), state).reshape(1, -1)
    
    
    # Predict value using one-hot data
    oneKnnPred = oneKnnModel.predict(oneData)
    onePolyPred = onePolyModel.predict(oneData)
    
    # Average the one-hot predictions
    oneAvgPred = (oneKnnPred + onePolyPred) / 2
    
    # Predict value using the median data
    medianKnnPred = medianKnnModel.predict(medianData)
    medianPolyPred = medianPolyModel.predict(medianData)
    
    # Average the median predictions
    medianAvgPred = (medianKnnPred + medianPolyPred) / 2
    
    
    print()
    
    # Display the one-hot predictions
    print("One-Hot predictions")
    print(f"KNN evaluation: {round(oneKnnPred[0], 2)}")
    print(f"Polynomial evaluation: {round(onePolyPred[0], 2)}")
    print(f"Average evaluation: {round(oneAvgPred[0], 2)}")

    print()
    
    # Display the median predictions
    print("Median predictions")
    print(f"KNN evaluation: {round(medianKnnPred[0], 2)}")
    print(f"Polynomial evaluation: {round(medianPolyPred[0], 2)}")
    print(f"Average evaluation: {round(medianAvgPred[0], 2)}")



predict()