import streamlit as st
import pickle
import pandas as pd

# Teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah',
    'Mohali', 'Bengaluru'
]

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set the page config
st.set_page_config(
    page_title="IPL Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        background-color: #1E3D59;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #1E3D59;
        color: white;
    }
    .metric-box {
        background-color: #A3CEF1;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 10px;
    }
    .details-box {
        background-color: #A3CEF1;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        color: black;
    }
    .details-box p {
        color: black;
    }
    .stButton>button {
        background-color: #1E3D59;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description of the app
st.markdown('<div class="main-title"><h1>IPL Win Predictor</h1></div>', unsafe_allow_html=True)
st.markdown("""
    This application predicts the win probability of an IPL team based on the current match situation.
""")

# Sidebar for inputs
st.sidebar.header('Match Inputs')

# Team selection
batting_team = st.sidebar.selectbox('Select the batting team', sorted(teams))
bowling_team = st.sidebar.selectbox('Select the bowling team', sorted(teams))

# City selection
selected_city = st.sidebar.selectbox('Select host city', sorted(cities))

# Target input
target = st.sidebar.number_input('Target', min_value=1, step=1)

# Score, overs, and wickets input
score = st.sidebar.number_input('Score', min_value=0, step=1)
overs = st.sidebar.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
wickets = st.sidebar.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Predict button
if st.sidebar.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Create input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict probability
    result = pipe.predict_proba(input_df)
    loss_prob = result[0][0]
    win_prob = result[0][1]

    # Display results with custom style
    st.markdown(
        f"""
        <div class="metric-box">
            <h2>Win Probability</h2>
            <h3 style="color: green;">{batting_team}: {win_prob * 100:.2f}%</h3>
            <h3 style="color: red;">{bowling_team}: {loss_prob * 100:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Additional insights
    st.markdown(
        """
        <div class="details-box">
            <h2>Match Details</h2>
            <p><b>Batting Team:</b> {batting_team}</p>
            <p><b>Bowling Team:</b> {bowling_team}</p>
            <p><b>City:</b> {selected_city}</p>
            <p><b>Target:</b> {target}</p>
            <p><b>Current Score:</b> {score}</p>
            <p><b>Overs Completed:</b> {overs}</p>
            <p><b>Wickets Out:</b> {wickets}</p>
            <p><b>Runs Left:</b> {runs_left}</p>
            <p><b>Balls Left:</b> {balls_left}</p>
            <p><b>Current Run Rate (CRR):</b> {crr:.2f}</p>
            <p><b>Required Run Rate (RRR):</b> {rrr:.2f}</p>
        </div>
        """.format(
            batting_team=batting_team,
            bowling_team=bowling_team,
            selected_city=selected_city,
            target=target,
            score=score,
            overs=overs,
            wickets=wickets,
            runs_left=runs_left,
            balls_left=balls_left,
            crr=crr,
            rrr=rrr
        ),
        unsafe_allow_html=True
    )
