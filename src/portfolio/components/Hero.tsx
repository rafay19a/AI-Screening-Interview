import { motion } from 'framer-motion';
import { ArrowUpRight } from 'lucide-react';

export const Hero = () => {
  return (
    <section className="min-h-screen flex flex-col justify-end pb-24 px-6 md:px-12 relative bg-black text-white">
      <div className="max-w-screen-2xl mx-auto w-full grid grid-cols-1 md:grid-cols-12 gap-12 items-end">

        <div className="md:col-span-8">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <span className="text-[10px] uppercase tracking-[0.4em] text-white/40 font-medium block mb-12">
              AI Engineer & Full Stack Developer
            </span>
            <h1 className="text-[12vw] md:text-[8vw] leading-[0.9] font-medium tracking-tighter-premium mb-12">
              Building the<br />Future of AI.
            </h1>
          </motion.div>
        </div>

        <div className="md:col-span-4 md:pl-12">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8, duration: 1 }}
            className="border-t border-white/10 pt-12"
          >
            <p className="text-lg text-white/50 leading-relaxed font-light mb-12">
              Based in Islamabad. Delivering high-stakes digital products that merge technical complexity with editorial design.
            </p>

            <div className="flex flex-wrap gap-8 items-center">
              <a href="https://github.com/rafay19a" target="_blank" rel="noreferrer" className="flex items-center gap-2 group text-[10px] uppercase tracking-widest font-bold">
                GitHub <ArrowUpRight className="w-3 h-3 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </a>
              <a href="https://www.linkedin.com/in/abdul-rafay-arshad-a39b1b229?utm_source=share_via&utm_content=profile&utm_medium=member_android" target="_blank" rel="noreferrer" className="flex items-center gap-2 group text-[10px] uppercase tracking-widest font-bold">
                LinkedIn <ArrowUpRight className="w-3 h-3 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </a>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Decorative vertical lines */}
      <div className="absolute top-0 left-0 w-px h-full bg-white/5 hidden md:block" />
      <div className="absolute top-0 right-0 w-px h-full bg-white/5 hidden md:block" />
      <div className="absolute top-0 left-1/4 w-px h-full bg-white/5 hidden md:block" />
    </section>
  );
};
