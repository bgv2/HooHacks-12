"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import './styles.css';

export default function Home() {
	const [contacts, setContacts] = useState<string[]>([""]);
	const [codeword, setCodeword] = useState("");
	const [session, setSession] = useState<any>(null);
	const [loading, setLoading] = useState(true);
	const router = useRouter();

	useEffect(() => {
		// Fetch session data from an API route
		fetch("/auth/session")
			.then((response) => response.json())
			.then((data) => {
				console.log("Session data received:", data);
				setSession(data.session);
				
				setLoading(false);
			})
			.catch((error) => {
				console.error("Failed to fetch session:", error);
				setLoading(false);
			});
	}, []);

	const handleInputChange = (index: number, value: string) => {
		const updatedContacts = [...contacts];
		updatedContacts[index] = value; // Update the specific input value
		setContacts(updatedContacts);
	};

	const addContactInput = () => {
		setContacts([...contacts, ""]); // Add a new empty input
	};

	function saveToDB() {
		alert("Saving contacts...");
		const contactInputs = document.querySelectorAll(
			".text-input"
		) as NodeListOf<HTMLInputElement>;
		const contactValues = Array.from(contactInputs).map((input) => input.value);

		fetch("/api/databaseStorage", {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				email: session?.user?.email || "",
				codeword: codeword,
				contacts: contactValues,
			}),
		})
			.then((response) => {
				if (response.ok) {
					// alert("Contacts saved successfully!");
				} else {
					alert("Error saving contacts.");
				}
			})
			.catch((error) => {
				console.error("Error:", error);
				alert("Error saving contacts.");
			});
	}

	if (loading) {
		return <div>Loading...</div>;
	}
	

	// If no session, show sign-up and login buttons
	if (!session) {
		return (
			<div className="space-y-7  items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
				
				<h1
  className="space-y-3 text-4xl text-lime-500 subpixel-antialiased animate-fadeIn text-center font-stretch-semi-expanded"
  style={{ animationDelay: "0s", fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }}
>
  welcome to Fauxcall
</h1>


<p
  className="text-xl text-gray-700"
  style={{
    animation: 'fadeIn 1s ease-in-out forwards',
    animationDelay: '1s',
    opacity: 0,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  }}
>
  We empower you to feel safe.
</p>

<p
  className="text-xl text-gray-700"
  style={{
    animation: 'fadeIn 1s ease-in-out forwards',
    animationDelay: '2s',
    opacity: 0,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  }}
>
  Whenever and wherever.
</p>


<div
  style={{
    animation: 'fadeIn 1s ease-in-out forwards',
    animationDelay: '3s',
    opacity: 0,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  }}
>
<div className="space-y-20 flex flex-col items-center">
  <a href="/auth/login?screen_hint=signup">
    <button className="p-4 h-16 w-32 mb-10 mt-4 text-xl text-green-400 border-2 border-violet-900 rounded-md transition-opacity duration-1000 opacity-0 animate-fadeIn delay-0">
      Sign up
    </button>
  </a>

  <p className="animate-fadeIn delay-1 opacity-0">
    Already have an account?
  </p>

  <a href="/auth/login">
    <button className="p-4 h-16 w-32 text-xl text-black border-2 border-gray-200 rounded-md transition-opacity duration-1000 opacity-0 animate-fadeIn delay-2">
      Log in
    </button>
  </a>
</div>



</div>
				

			</div>
		);
	}

	return (
		
		<div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-10 sm:p-20 font-[family-name:var(--font-geist-sans)]">
			<main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
			

				<h1 className="space-y-3 text-2xl text-lime-500 subpixel-antialiased font-stretch-semi-expanded font-sans">welcome to <span className="font-bold text-2xl text-lime-700">Fauxcall</span>, {session.user.nickname}!</h1>
				
				<p>
					To begin, set a codeword and emergency contacts. If you stop speaking or say the codeword, these contacts will be
					notified.
				</p>
				{/* form for setting codeword */}
				<form
					className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start"
					onSubmit={(e) => e.preventDefault()}
				>
					<input
						type="text"
						value={codeword}
						onChange={(e) => setCodeword(e.target.value)}
						placeholder="Codeword"
						className="border border-gray-300 rounded-md p-2"
					/>
					{/* <button
						className="bg-blue-500 text-white font-semibold font-lg rounded-md p-2"
						type="submit"
					>
						Set codeword
					</button> */}
				</form>
				{/* form for adding contacts */}
				<form
					id="Contacts"
					className="space-y-5 flex flex-col gap-[32px] row-start-2 items-center sm:items-start"
					onSubmit={(e) => e.preventDefault()}
				>
					{contacts.map((contact, index) => (
						<input
							key={index}
							type="text"
							value={contact}
							onChange={(e) => handleInputChange(index, e.target.value)}
							placeholder={`Contact ${index + 1}`}
							className="border border-gray-300 rounded-md p-2"
						/>
					))}
					<button
						onClick={addContactInput}
						className="bg-emerald-500 text-white
						font-semibold font-lg rounded-md p-2"
						type="button"
					>
						Add Contact
					</button>
					<hr />
					<button
						type="button"
						onClick={saveToDB}
						className="bg-slate-500 text-yellow-300 text-stretch-50% font-lg rounded-md p-2"
					>
						Save Settings
					</button>
				</form>
				<div>
					<a href="/call">
						<button className="bg-zinc-700 text-lime-300 font-semibold font-lg rounded-md p-2">
							Call
						</button>
					</a>
				</div>
				<p>
					<a href="/auth/logout" className="font-semibold font-lg rounded-md p-2">
						<button>Log out</button>
					</a>
				</p>
			</main>
		</div>
	);
}
