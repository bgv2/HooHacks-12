"use client";
import { useState } from "react";
import { auth0 } from "../lib/auth0";

export default async function Home() {
	const [contacts, setContacts] = useState<string[]>([]);
	const [codeword, setCodeword] = useState("");

	const session = await auth0.getSession();

	console.log("Session:", session?.user);

	// If no session, show sign-up and login buttons
	if (!session) {	

		return (
			<div className="items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
				<main className="flex flex-row gap-[32px] row-start-2 items-center sm:items-start">
					<a href="/auth/login?screen_hint=signup">
						<button className="w-16 text-green-600">Sign up</button>
					</a>
					<a href="/auth/login">
						<button className = "w-16 text-green-600">Log in</button>
					</a>
				</main>
				<h1>fauxcall</h1>
				<h2>set emergency contacts</h2>
				<p>if you stop speaking or say the codeword, these contacts will be notified</p>
				{/* form for setting codeword */}
				<form className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start" onSubmit={(e) => e.preventDefault()}>
					<input
						type="text"
						value={codeword}
						onChange={(e) => setCodeword(e.target.value)}
						placeholder="codeword"
						className="border border-gray-300 rounded-md p-2"
					/>
					<button
					className="bg-blue-500 text-white rounded-md p-2"
					type="submit">Set codeword</button>
				</form>
				{/* form for adding contacts */}
				<form className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start" onSubmit={(e) => e.preventDefault()}>
					<input
						type="text"
						value={contacts}
						onChange={(e) => setContacts(e.target.value.split(","))}
						placeholder="contacts (comma separated)"
						className="border border-gray-300 rounded-md p-2"
					/>
					<button type="submit">Set contacts</button>
				</form>
			</div>
		);
	}

	return (
		<div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
			<main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
				<h1>Welcome, {session.user.name}!</h1>
				<p>
					<a href="/auth/logout">
						<button>Log out</button>
					</a>
				</p>
			</main>
		</div>
	);
}
