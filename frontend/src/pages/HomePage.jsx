// ROLLBACK: Original was lowercase '../components/layout/Header' and '../components/layout/Footer'
// Changed to match actual folder name 'Layout' (uppercase) for Linux compatibility
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import { Hero, Features, Testimonials, CTA } from '../components/sections';

export default function HomePage() {
  return (
    <div className="relative flex min-h-screen flex-col bg-background">
      <Header />
      <main className="flex-1">
        <Hero />
        <Features />
        <Testimonials />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}
