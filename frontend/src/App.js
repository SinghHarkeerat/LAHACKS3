import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import HomePage from './components/HomePage';
import LipPage from './components/Lip';
import ASLPage from './components/ASL';
import ExitPage from './components/ExitPage';

function App() {
    return (
        <Router>
            <Switch>
                <Route path="/" exact component={HomePage} />
                <Route path="/deaf" component={LipPage} />
                <Route path="/mute" component={ASLPage} />
                <Route path="/exit" component={ExitPage} />
            </Switch>
        </Router>
    );
}

export default App;