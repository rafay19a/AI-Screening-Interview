import { motion } from 'framer-motion';

const stats = [
  { label: "AI Workload Reduction", value: "60%" },
  { label: "Systems Built", value: "10+" },
  { label: "Performance Boost", value: "2.4x" },
  { label: "Users Impacted", value: "5K+" }
];

export const Impact = () => {
  return (
    <section className="py-48 bg-black px-6 md:px-12 border-t border-white/5">
      <div className="max-w-screen-2xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-12">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.label}
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.1, duration: 1 }}
            className="text-center md:text-left"
          >
            <span className="text-[10px] uppercase tracking-[0.4em] text-white/30 block mb-6">{stat.label}</span>
            <span className="text-6xl md:text-8xl font-medium tracking-tighter-premium">{stat.value}</span>
          </motion.div>
        ))}
      </div>
    </section>
  );
};
