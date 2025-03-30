'use client';

import React from 'react';

const CallPage = () => {
  const [callDuration, setCallDuration] = React.useState(0);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setCallDuration((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

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
        <button onClick={handleEmergency} className="bg-red-500 text-white rounded-md p-2">Emergency</button>
        <button className="bg-blue-500 text-white rounded-md p-2"
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