import { auth0 } from "../lib/auth0";

export default async function Home() {
	const session = await auth0.getSession();

	console.log("Session:", session?.user);

	// If no session, show sign-up and login buttons
	if (!session) {	

		return (
			<div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
				<main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
					<a href="/auth/login?screen_hint=signup">
						<button>Sign up</button>
					</a>
					<a href="/auth/login">
						<button>Log in</button>
					</a>
				</main>
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
