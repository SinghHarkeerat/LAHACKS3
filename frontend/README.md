# React-Python Website

This project is a web application that combines a Flask backend with a React frontend. It features pages for "Deaf" and "Mute," each with live camera functionality, as well as a home page and an exit page.

## Project Structure

- **backend/**: Contains the Flask backend code.
  - **app.py**: Main entry point for the Flask application.
  - **requirements.txt**: Lists the dependencies required for the backend.

- **frontend/**: Contains the React frontend code.
  - **public/**: Contains static files.
    - **index.html**: Main HTML file for the React application.
  - **src/**: Contains the source code for the React application.
    - **App.js**: Main component that sets up routing.
    - **components/**: Contains individual React components.
      - **Camera.js**: Component for accessing the user's webcam.
      - **DeafPage.js**: Component for the Deaf page.
      - **HomePage.js**: Component for the home page.
      - **MutePage.js**: Component for the Mute page.
      - **ExitPage.js**: Component for the exit page.
    - **index.css**: CSS styles for the application.
    - **index.js**: Entry point for the React application.
  - **package.json**: Configuration file for npm.

## Getting Started

1. **Clone the repository**:
   ```
   git clone <repository-url>
   ```

2. **Navigate to the backend directory** and install the required dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```

3. **Run the Flask backend**:
   ```
   python app.py
   ```

4. **Navigate to the frontend directory** and install the required dependencies:
   ```
   cd frontend
   npm install
   ```

5. **Run the React application**:
   ```
   npm start
   ```

## Features

- Home page with navigation buttons to Deaf and Mute pages.
- Live camera feed on both Deaf and Mute pages.
- Exit page to indicate the user has exited the application.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.