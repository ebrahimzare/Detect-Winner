import React from 'react';
// import logo from './logo.svg';
 import './App.css';

function App() {
  return (
    <div className="App">
      <div id="container">
          <div id="vid_container">
           <video id="video" autoPlay playsInline />
          
           <div id="video_overlay" />

        
          </div>

      


        <div id="gui_controls">
          <button id="switchCameraButton" name="switch Camera" type="button" aria-pressed="false" />
          <button id="takePhotoButton" name="take Photo" type="button" />
          <button id="toggleFullScreenButton" name="toggle FullScreen" type="button" aria-pressed="false" />
        </div>


      </div>
    </div>
  );
}

export default App;
