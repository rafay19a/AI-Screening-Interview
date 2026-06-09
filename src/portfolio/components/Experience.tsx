import { motion } from 'framer-motion';

const experiences = [
  {
    title: "Data Science Lead Intern",
    company: "Ram Burners",
    period: "Mar — May 2025",
    description: "Developed automated data processing workflows and predictive ML models to drive executive decision-making."
  },
  {
    title: "AI Systems Intern",
    company: "InternCraft",
    period: "Jul — Sep 2025",
    description: "Engineered production-level data pipelines and high-accuracy dashboards for real-world client datasets."
  },
  {
    title: "Technical Content Strategist",
    company: "Spreadcheaters",
    period: "Sep — Dec 2024",
    description: "Crafted SEO-driven technical architecture narratives and market expansion strategies."
  }
];

export const Experience = () => {
  return (
    <section id="experience" className="py-48 bg-black px-6 md:px-12 border-t border-white/5">
      <div className="max-w-screen-2xl mx-auto">
        <h2 className="text-5xl font-medium tracking-tighter-premium mb-32">Experience.</h2>

        <div className="space-y-px">
          {experiences.map((exp, index) => (
            <motion.div
              key={`${exp.company}-${index}`}
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1 }}
              className="group border-t border-white/10 py-16 grid grid-cols-1 md:grid-cols-12 gap-12 items-baseline hover:bg-white/[0.02] transition-colors px-4 -mx-4"
            >
              <div className="md:col-span-2 text-[10px] uppercase tracking-[0.3em] text-white/30">{exp.period}</div>
              <div className="md:col-span-4 text-3xl tracking-tight font-medium group-hover:pl-4 transition-all duration-500">{exp.company}</div>
              <div className="md:col-span-3 text-white/50 italic font-light">{exp.title}</div>
              <div className="md:col-span-3 text-sm text-white/40 leading-relaxed font-light">{exp.description}</div>
            </motion.div>
          ))}
          <div className="border-t border-white/10" />
        </div>
      </div>
    </section>
  );
};
