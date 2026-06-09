import { motion } from 'framer-motion';

const experiences = [
  {
    title: "Data Science Intern",
    company: "Ram Burners",
    period: "2025",
    description: "Architected predictive ML models and automated data ingestion pipelines using SQL and Python."
  },
  {
    title: "AI Solutions Intern",
    company: "InternCraft",
    period: "2025",
    description: "Engineered high-accuracy analytical dashboards and optimized production-level data preprocessing."
  },
  {
    title: "Technical Content Strategist",
    company: "Spreadcheaters",
    period: "2024",
    description: "Developed SEO-driven technical documentation and content alignment strategies for digital growth."
  }
];

export const Experience = () => {
  return (
    <section className="py-32 bg-black px-4 border-t border-white/5">
      <div className="max-w-4xl mx-auto">
        <motion.div
           initial={{ opacity: 0, y: 20 }}
           whileInView={{ opacity: 1, y: 0 }}
           viewport={{ once: true }}
           className="mb-24"
        >
          <h2 className="text-4xl md:text-5xl font-medium text-white mb-6 tracking-tighter">Experience</h2>
        </motion.div>

        <div className="space-y-32">
          {experiences.map((exp, index) => (
            <motion.div
              key={`${exp.company}-${index}`}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1, duration: 1, ease: [0.16, 1, 0.3, 1] }}
              className="flex flex-col md:flex-row gap-8 md:gap-24"
            >
              <div className="md:w-1/4">
                <span className="text-[10px] uppercase tracking-[0.3em] text-white/20 font-bold block mb-2">{exp.period}</span>
                <h3 className="text-white font-medium tracking-tight uppercase text-xs">{exp.company}</h3>
              </div>

              <div className="md:w-3/4">
                <h4 className="text-2xl font-medium text-white mb-6 tracking-tight">{exp.title}</h4>
                <p className="text-white/40 leading-relaxed font-light max-w-xl text-lg">
                  {exp.description}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
