import { motion } from 'framer-motion';
import { Mail, ArrowDown } from 'lucide-react';

const GithubIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
    <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
    <path d="M9 18c-4.51 2-5-2-7-2" />
  </svg>
);

const LinkedinIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-5 h-5">
    <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z" />
    <rect width="4" height="12" x="2" y="9" />
    <circle cx="4" cy="4" r="2" />
  </svg>
);

export const Hero = () => {
  return (
    <section className="min-h-screen flex flex-col justify-center items-center relative overflow-hidden bg-black text-white px-4">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
        className="z-10 text-center max-w-4xl"
      >
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="px-4 py-1 rounded-full border border-white/10 bg-white/5 text-white/50 text-xs tracking-widest uppercase mb-8 inline-block"
        >
          Based in Islamabad, Pakistan
        </motion.span>

        <h1 className="text-6xl md:text-8xl font-medium mb-8 tracking-tighter text-white">
          Abdul Rafay Arshad
        </h1>

        <p className="text-lg md:text-xl text-white/60 mb-12 leading-relaxed max-w-2xl mx-auto font-light">
          Crafting intelligent AI systems and high-performance web experiences with minimalist precision.
        </p>

        <div className="flex flex-wrap justify-center gap-6 mb-16">
          <motion.a
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            href="mailto:rafay19a@gmail.com"
            className="flex items-center gap-2 px-8 py-4 bg-white text-black rounded-full font-medium transition-all"
          >
            <Mail className="w-4 h-4" />
            Get in touch
          </motion.a>

          <div className="flex gap-4 items-center">
             <motion.a
                whileHover={{ y: -2, color: '#fff' }}
                href="https://github.com/rafay19a"
                target="_blank"
                rel="noreferrer"
                className="p-4 rounded-full border border-white/10 text-white/40 transition-colors"
                title="GitHub"
              >
              <GithubIcon />
            </motion.a>
            <motion.a
                whileHover={{ y: -2, color: '#fff' }}
                href="https://www.linkedin.com/in/abdul-rafay-arshad-a39b1b229?utm_source=share_via&utm_content=profile&utm_medium=member_android"
                target="_blank"
                rel="noreferrer"
                className="p-4 rounded-full border border-white/10 text-white/40 transition-colors"
                title="LinkedIn"
              >
              <LinkedinIcon />
            </motion.a>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-12 left-1/2 -translate-x-1/2 text-white/20 flex flex-col items-center gap-4"
      >
        <span className="text-[10px] uppercase tracking-[0.3em] font-medium">Discover Work</span>
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
        >
          <ArrowDown className="w-4 h-4" />
        </motion.div>
      </motion.div>
    </section>
  );
};
