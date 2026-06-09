import { motion } from 'framer-motion';
import { Mail, Globe, User, ArrowDown } from 'lucide-react';

export const Hero = () => {
  return (
    <section className="min-h-screen flex flex-col justify-center items-center relative overflow-hidden bg-slate-950 text-white px-4">
      {/* Background decoration */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[120px]" />
        <div className="absolute -bottom-[10%] -right-[10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[120px]" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="z-10 text-center max-w-4xl"
      >
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="px-4 py-1.5 rounded-full border border-slate-800 bg-slate-900/50 text-slate-400 text-sm font-medium mb-6 inline-block"
        >
          Available for new opportunities
        </motion.span>

        <h1 className="text-5xl md:text-7xl font-bold mb-6 tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-white to-slate-400">
          Abdul Rafay Arshad
        </h1>

        <p className="text-xl md:text-2xl text-slate-400 mb-10 leading-relaxed font-light">
          AI Engineer & Full Stack Developer building <br className="hidden md:block" />
          <span className="text-white font-normal">intelligent automation</span> and <span className="text-white font-normal">scalable web solutions</span>.
        </p>

        <div className="flex flex-wrap justify-center gap-4 mb-12">
          <a
            href="mailto:rafay19a@gmail.com"
            className="flex items-center gap-2 px-6 py-3 bg-white text-slate-950 rounded-full font-semibold hover:bg-slate-200 transition-colors"
          >
            <Mail className="w-5 h-5" />
            Contact Me
          </a>
          <div className="flex gap-4 items-center">
             <a href="#" className="p-3 rounded-full border border-slate-800 hover:bg-slate-900 transition-colors" title="GitHub">
              <Globe className="w-5 h-5" />
            </a>
            <a href="https://www.linkedin.com/in/abdul-rafay-arshad-63567a21b" className="p-3 rounded-full border border-slate-800 hover:bg-slate-900 transition-colors" title="LinkedIn">
              <User className="w-5 h-5" />
            </a>
          </div>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 1 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 text-slate-500 flex flex-col items-center gap-2"
      >
        <span className="text-xs uppercase tracking-widest font-medium">Scroll to explore</span>
        <motion.div
          animate={{ y: [0, 5, 0] }}
          transition={{ repeat: Infinity, duration: 2 }}
        >
          <ArrowDown className="w-4 h-4" />
        </motion.div>
      </motion.div>
    </section>
  );
};
