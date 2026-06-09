import { motion } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';

export const Hero = () => {
  return (
    <section className="min-h-screen flex flex-col justify-center px-6 md:px-12 lg:px-24 relative bg-black text-white pt-32">
      <div className="max-w-screen-2xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-24 items-start">

        <div className="lg:col-span-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <span className="text-[10px] uppercase tracking-[0.4em] text-white/40 font-medium block mb-12">
              AI Engineer | Data Scientist | Prompt Engineer | Full Stack Developer
            </span>
            <h1 className="text-[12vw] lg:text-[7vw] leading-[0.9] font-medium tracking-tighter-premium mb-16">
              Abdul Rafay<br />Arshad.
            </h1>
          </motion.div>
        </div>

        <div className="lg:col-span-4 lg:pt-48">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8, duration: 1 }}
            className="border-t border-white/10 pt-12"
          >
            <h2 className="text-[10px] uppercase tracking-[0.3em] text-white/30 mb-8 font-bold">Professional Summary</h2>
            <p className="text-xl text-white/60 leading-relaxed font-light mb-12 italic">
              "AI Engineer and Data Scientist with hands-on experience building AI-powered systems, full-stack web applications, and intelligent automation solutions. Skilled in machine learning, NLP, and prompt engineering, with a strong focus on developing scalable, real-world products."
            </p>

            <div className="flex flex-wrap gap-8 items-center">
              <a href="https://github.com/rafay19a" target="_blank" rel="noreferrer" className="flex items-center gap-2 group text-[10px] uppercase tracking-widest font-bold border-b border-white/10 pb-1 hover:border-white transition-colors">
                GitHub <ArrowUpRight className="w-3 h-3 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </a>
              <a href="https://www.linkedin.com/in/abdul-rafay-arshad-a39b1b229?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" rel="noreferrer" className="flex items-center gap-2 group text-[10px] uppercase tracking-widest font-bold border-b border-white/10 pb-1 hover:border-white transition-colors">
                LinkedIn <ArrowUpRight className="w-3 h-3 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </a>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Decorative background branding */}
      <div className="absolute bottom-12 left-6 md:left-12 lg:left-24 opacity-5 pointer-events-none select-none">
         <span className="text-[15vw] font-bold tracking-tighter leading-none">ARSHAD</span>
      </div>

      {/* Decorative vertical lines */}
      <div className="absolute top-0 left-0 w-px h-full bg-white/5 hidden md:block" />
      <div className="absolute top-0 right-0 w-px h-full bg-white/5 hidden md:block" />
    </section>
  );
};
