import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to load initial graph and dataset
def load_initial_graph(dataset, ax):
    if dataset == "Binary":
        X, y = make_blobs(n_features=2, centers=2, random_state=6)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y
    elif dataset == "Multiclass":
        X, y = make_blobs(n_features=2, centers=3, random_state=2)
        ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
        return X, y

# Function to draw meshgrid for decision boundary
def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)

    XX, YY = np.meshgrid(a, b)

    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

# Streamlit setup for welcome page
def welcome_page():
    st.title("Welcome to the Logistic Regression Classifier App!")
    st.markdown("""
    ### Made by: Zunair Zafar
    #### Copyright © 2025
    - This app allows you to interactively explore the behavior of a Logistic Regression Classifier on various datasets.
    - You can select different configurations and view the decision boundary and model accuracy.
    
    --- 

    Press the **Enter** button below to begin exploring the app.
    """)

    # Enter button to navigate to the main app
    if st.button("Enter"):
        st.session_state.page = "main"  # Set the session state to 'main' page

# Main page for the Logistic Regression Classifier
def main_page():
    st.sidebar.markdown("# Logistic Regression Classifier")

    # Sidebar options with input validation
    dataset = st.sidebar.selectbox(
        'Select Dataset',
        ('Binary', 'Multiclass')
    )

    penalty = st.sidebar.selectbox(
        'Regularization',
        ('l2', 'l1', 'elasticnet', 'none')
    )

    # Validate 'C' input (must be a positive float)
    c_input = st.sidebar.number_input('C', value=1.0, min_value=0.01, step=0.01)

    solver = st.sidebar.selectbox(
        'Solver',
        ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
    )

    # Validate 'max_iter' input (must be a positive integer)
    max_iter = st.sidebar.number_input('Max Iterations', value=100, min_value=1, step=1)

    multi_class = st.sidebar.selectbox(
        'Multi Class',
        ('auto', 'ovr', 'multinomial')
    )

    # Validate 'l1_ratio' input (must be between 0 and 1)
    l1_ratio = st.sidebar.number_input('l1 Ratio', value=0.5, min_value=0.0, max_value=1.0, step=0.05)

    # Ensure that all inputs are valid and handle errors gracefully
    if not (0.0 <= l1_ratio <= 1.0):
        st.error("Invalid L1 ratio. Please enter a value between 0 and 1.")
    else:
        # Load initial graph
        fig, ax = plt.subplots()

        # Plot initial graph
        X, y = load_initial_graph(dataset, ax)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
        orig = st.pyplot(fig)

        if st.sidebar.button('Run Algorithm'):
            try:
                orig.empty()  # Clear the previous plot

                # Logistic Regression classifier
                clf = LogisticRegression(
                    penalty=penalty, C=c_input, solver=solver, max_iter=max_iter,
                    multi_class=multi_class, l1_ratio=l1_ratio
                )
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                XX, YY, input_array = draw_meshgrid(X)
                labels = clf.predict(input_array)

                # Plot decision boundary
                ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
                plt.xlabel("Col1")
                plt.ylabel("Col2")
                orig = st.pyplot(fig)

                # Display accuracy
                st.subheader(f"Accuracy for Logistic Regression: {round(accuracy_score(y_test, y_pred), 2)}")

            except Exception as e:
                st.error(f"An error occurred while running the algorithm: {str(e)}")

# Main app logic
if __name__ == "__main__":
    # Check if the session state has the page variable
    if 'page' not in st.session_state:
        st.session_state.page = 'welcome'  # Default to welcome page

    # Render the appropriate page
    if st.session_state.page == 'welcome':
        welcome_page()
    elif st.session_state.page == 'main':
        main_page()
