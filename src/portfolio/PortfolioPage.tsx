import { Hero } from './components/Hero';
import { Projects } from './components/Projects';
import { Skills } from './components/Skills';
import { Experience } from './components/Experience';
import { Mail, Globe, User } from 'lucide-react';

const PortfolioPage = () => {
  return (
    <div className="bg-slate-950 min-h-screen text-slate-200 selection:bg-blue-500/30 selection:text-blue-200">
      <Hero />
      <div className="relative">
        {/* Glow effects between sections */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-4xl h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent" />
        <Projects />
        <Skills />
        <Experience />
      </div>

      <footer className="py-20 border-t border-slate-900 bg-slate-950 px-4">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-10">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">Abdul Rafay Arshad</h2>
            <p className="text-slate-500">AI Engineer & Full Stack Developer</p>
          </div>

          <div className="flex flex-wrap justify-center gap-8">
             <a href="mailto:rafay19a@gmail.com" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
              <Mail className="w-5 h-5" />
              Email
            </a>
            <a href="https://www.linkedin.com/in/abdul-rafay-arshad-63567a21b" target="_blank" rel="noreferrer" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
              <User className="w-5 h-5" />
              LinkedIn
            </a>
            <a href="#" className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
              <Globe className="w-5 h-5" />
              GitHub
            </a>
          </div>

          <div className="text-slate-600 text-sm">
            © {new Date().getFullYear()} • Built with React & Framer Motion
          </div>
        </div>
      </footer>
    </div>
  );
};

export default PortfolioPage;
