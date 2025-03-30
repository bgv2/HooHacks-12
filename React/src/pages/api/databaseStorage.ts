import { NextApiRequest, NextApiResponse } from "next";
import mongoose from "mongoose";

const uri = process.env.MONGODB_URI || "mongodb://localhost:27017/mydatabase";
const clientOptions = { serverApi: { version: "1" as const, strict: true, deprecationErrors: true } };

// Create a reusable connection function
async function connectToDatabase() {
  if (mongoose.connection.readyState === 0) {
    // Only connect if not already connected
    await mongoose.connect(uri, clientOptions);
    console.log("Connected to MongoDB!");
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // Ensure the database is connected
    await connectToDatabase();

    if (req.method === 'POST') {
      const { codeword, contacts } = req.body;

      // Perform database operations here
      console.log("Codeword:", codeword);
      console.log("Contacts:", contacts);

      res.status(200).json({ success: true, message: "Data saved successfully!" });
    } else {
      res.setHeader('Allow', ['POST']);
      res.status(405).end(`Method ${req.method} Not Allowed`);
    }
  } catch (error) {
    console.error("Error:", error);
    res.status(500).json({ success: false, error: "Internal Server Error" });
  }
}