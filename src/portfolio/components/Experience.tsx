import { motion } from 'framer-motion';
import { Briefcase, Calendar } from 'lucide-react';

const experiences = [
  {
    title: "Data Science Intern",
    company: "Ram Burners",
    period: "March 2025 - May 2025",
    description: [
      "Analyzed large-scale datasets to identify patterns, improving business decision-making efficiency.",
      "Built machine learning models that enhanced predictive accuracy and performance.",
      "Automated data processing workflows using Python and SQL, reducing manual effort."
    ]
  },
  {
    title: "Data Science Intern",
    company: "InternCraft",
    period: "July 2025 - September 2025",
    description: [
      "Collected, cleaned, and preprocessed large datasets for analysis and modeling.",
      "Built and optimized machine learning models to improve accuracy.",
      "Designed dashboards and reports to support business decisions."
    ]
  },
  {
    title: "Content Writer",
    company: "Spreadcheaters",
    period: "September 2024 - December 2024",
    description: [
      "Created SEO-optimized content for blogs and websites, improving engagement.",
      "Conducted keyword research and trend analysis for content strategy."
    ]
  }
];

export const Experience = () => {
  return (
    <section className="py-24 bg-slate-950 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Professional Experience</h2>
        </div>

        <div className="space-y-12">
          {experiences.map((exp, index) => (
            <motion.div
              key={`${exp.company}-${index}`}
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="relative pl-8 border-l border-slate-800"
            >
              {/* Timeline Dot */}
              <div className="absolute left-[-5px] top-0 w-[10px] h-[10px] rounded-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.5)]" />

              <div className="flex flex-wrap justify-between items-start gap-4 mb-4">
                <div>
                  <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <Briefcase className="w-5 h-5 text-slate-500" />
                    {exp.title}
                  </h3>
                  <p className="text-blue-400 font-medium">{exp.company}</p>
                </div>
                <div className="flex items-center gap-2 text-slate-500 text-sm font-medium">
                  <Calendar className="w-4 h-4" />
                  {exp.period}
                </div>
              </div>

              <ul className="space-y-3">
                {exp.description.map((item, i) => (
                  <li key={i} className="text-slate-400 leading-relaxed flex gap-3">
                    <span className="text-slate-600 mt-1.5">•</span>
                    {item}
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
