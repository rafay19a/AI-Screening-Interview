import { motion } from 'framer-motion';

export const Navbar = () => {
  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
      className="fixed top-0 left-0 w-full z-[100] px-6 py-8 flex justify-between items-baseline mix-blend-difference"
    >
      <div className="text-xl font-medium tracking-tighter">AR.</div>
      <div className="flex gap-12 text-[10px] uppercase tracking-[0.2em] font-medium">
        <a href="#projects" className="hover:text-white/50 transition-colors">Projects</a>
        <a href="#capabilities" className="hover:text-white/50 transition-colors">Capabilities</a>
        <a href="#experience" className="hover:text-white/50 transition-colors">Experience</a>
      </div>
      <a href="mailto:rafay19a@gmail.com" className="px-5 py-2 border border-white rounded-full text-[10px] uppercase tracking-widest hover:bg-white hover:text-black transition-all">
        Contact
      </a>
    </motion.nav>
  );
};
