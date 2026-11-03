import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Depression Severity Text Classifier",
  description:
    "Explainable NLP portfolio application using TF-IDF and Balanced Logistic Regression.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
