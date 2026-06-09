import { Navbar } from './components/Navbar';
import { Hero } from './components/Hero';
import { Impact } from './components/Impact';
import { Projects } from './components/Projects';
import { Skills } from './components/Skills';
import { Experience } from './components/Experience';
import { CustomCursor } from './components/CustomCursor';

const PortfolioPage = () => {
  return (
    <div className="bg-black min-h-screen text-white selection:bg-white selection:text-black cursor-none overflow-x-hidden font-sans">
      <CustomCursor />
      <Navbar />
      <Hero />
      <Impact />
      <div className="relative">
        <Projects />
        <Skills />
        <Experience />
      </div>

      <footer className="py-48 border-t border-white/5 bg-black px-6 md:px-12 relative overflow-hidden">
        {/* Background text decoration */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-[30vw] font-bold text-white/[0.02] tracking-tighter-premium pointer-events-none select-none">
          RAFAY
        </div>

        <div className="max-w-screen-2xl mx-auto flex flex-col md:flex-row justify-between items-end gap-24 relative z-10">
          <div>
            <h2 className="text-[10vw] md:text-[6vw] font-medium text-white mb-12 tracking-tighter-premium leading-none">Let's build<br />the future.</h2>
            <a href="mailto:rafay19a@gmail.com" className="text-2xl md:text-4xl font-light text-white/40 hover:text-white transition-colors duration-500 border-b border-white/10 pb-4">
              rafay19a@gmail.com
            </a>
          </div>

          <div className="flex flex-col items-end gap-12">
            <div className="flex gap-12">
              <a href="https://www.linkedin.com/in/abdul-rafay-arshad-a39b1b229?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" rel="noreferrer" className="text-[10px] uppercase tracking-[0.4em] text-white/40 hover:text-white transition-colors">
                LinkedIn
              </a>
              <a href="https://github.com/rafay19a" target="_blank" rel="noreferrer" className="text-[10px] uppercase tracking-[0.4em] text-white/40 hover:text-white transition-colors">
                GitHub
              </a>
            </div>

            <div className="text-white/20 text-[10px] uppercase tracking-[0.3em] font-medium">
              © {new Date().getFullYear()} • ARSHAD — ALL RIGHTS RESERVED
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default PortfolioPage;
