'use client';

import React, { useState, useRef } from 'react';

function CallPage() {
  const [callDuration, setCallDuration] = React.useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null >(null);
  const [audioBlob, setAudioBlob] = useState<Blob | null >(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null); // Reference for the MediaRecorder instance
  const audioChunks = useRef<Blob[]>([]); // Array to store audio data chunks
 
  
  React.useEffect(() => {
    const interval = setInterval(() => {
      setCallDuration((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);
  

  // Start recording
  const startRecording = async () => {
    try {
      console.log("Start recording function called");
      // permissions
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      // Collect audio data in chunks
      mediaRecorderRef.current.ondataavailable = (event) => {
        console.log("ondataavailable triggered, chunk size:", event.data.size);
        audioChunks.current.push(event.data);

      };

      // Handle stop event
      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunks.current, { type: 'audio/mp4' });
        console.log("Total chunks received:", audioChunks.current.length);
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioBlob(audioBlob);
        console.log('media recorder type', mediaRecorderRef.current.mimeType)
        console.log(audioBlob);
        setAudioUrl(audioUrl);
        audioChunks.current = []; 
      };

      mediaRecorderRef.current.start(1000);
      console.log("Recording started");
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing microphone:", err);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Play the recorded audio
  const playAudio = () => {
    const audio = new Audio(audioUrl);
    audio.play();
  };


  const handleEmergency = async () => {
		// send texts
		const response = await fetch("/api/sendMessage", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				message: `John Smith needs help.`,
			}),
		});

		if (!response.ok) {
			console.error("Error sending message:", response.statusText);
			return;
		}
	}

  return (
    <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">


        <h1 className="text-4xl font-bold text-center sm:text-left">
          Call with a Definitely Real Person</h1>



        <div className="text-2xl font-bold">{callDuration}s</div>

        {isRecording ? (
          <button onClick={stopRecording}>Stop Recording</button>
        ) : (
          <button onClick={startRecording}>Start Recording</button>
        )}
        {audioUrl && (
          <>
            <h2>Recorded Audio</h2>
            <audio controls src={audioUrl}></audio>
            <button onClick={playAudio}>Play Audio</button>
          </>
        )}
        <button onClick={handleEmergency} className="bg-red-500 text-white rounded-md p-8 text-2xl font-bold">Emergency</button>
        <button className="bg-blue-500 text-white rounded-md p-8 text-2xl font-bold"
          onClick={() => {
            window.location.href = '/';
          }}>
          End
        </button>
      </main>
    </div>
  );
};

export default CallPage;