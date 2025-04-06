import React from 'react';
import { useHistory } from 'react-router-dom';

const HomePage = () => {
    const history = useHistory();

    const navigateToLip = () => {
        history.push('/Lip');
    };

    const navigateToASL = () => {
        history.push('/ASL');
    };

    const exitApplication = () => {
        history.push('/exit');
    };

    return (
        <div>
            <h1>Welcome to the React-Python Website!</h1>
            <button onClick={navigateToDeaf}>Go to Deaf Page</button>
            <button onClick={navigateToMute}>Go to Mute Page</button>
            <button onClick={exitApplication}>Exit</button>
        </div>
    );
};

export default HomePage;