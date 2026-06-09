import { motion } from 'framer-motion';
import { BrainCircuit, Code2, Database, Terminal, Zap } from 'lucide-react';

const skillCategories = [
  {
    title: "AI & Machine Learning",
    icon: <BrainCircuit className="w-5 h-5 text-blue-400" />,
    skills: ["NLP", "LLMs", "Prompt Engineering", "Semantic Search", "Embeddings", "Whisper", "OpenAI API"],
    className: "md:col-span-2 md:row-span-2"
  },
  {
    title: "Full Stack Development",
    icon: <Code2 className="w-5 h-5 text-purple-400" />,
    skills: ["React", "JavaScript", "TypeScript", "Python", "Streamlit", "REST APIs"],
    className: "md:col-span-2 md:row-span-1"
  },
  {
    title: "Databases",
    icon: <Database className="w-5 h-5 text-emerald-400" />,
    skills: ["MySQL", "SQLite", "Supabase", "SQL"],
    className: "md:col-span-1 md:row-span-1"
  },
  {
    title: "Data Science",
    icon: <Terminal className="w-5 h-5 text-amber-400" />,
    skills: ["Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn"],
    className: "md:col-span-1 md:row-span-2"
  },
  {
    title: "Tools & Others",
    icon: <Zap className="w-5 h-5 text-rose-400" />,
    skills: ["Git", "GitHub", "SEO", "Automation"],
    className: "md:col-span-1 md:row-span-1"
  }
];

export const Skills = () => {
  return (
    <section className="py-24 bg-slate-950 px-4 relative">
      <div className="max-w-6xl mx-auto">
        <div className="mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Technical Expertise</h2>
          <p className="text-slate-400 max-w-2xl">
            My skill set bridges the gap between sophisticated AI models and high-performance web applications.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 auto-rows-[160px]">
          {skillCategories.map((cat, index) => (
            <motion.div
              key={cat.title}
              initial={{ opacity: 0, scale: 0.95 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.05 }}
              className={`p-6 rounded-3xl border border-slate-800 bg-slate-900/40 hover:border-slate-700 transition-colors flex flex-col justify-between ${cat.className}`}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-xl bg-slate-800 border border-slate-700">
                  {cat.icon}
                </div>
                <h3 className="font-bold text-white">{cat.title}</h3>
              </div>

              <div className="flex flex-wrap gap-2 mt-auto">
                {cat.skills.map(skill => (
                  <span key={skill} className="text-xs px-2 py-1 rounded-md bg-slate-800/50 text-slate-300 border border-slate-700/50">
                    {skill}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
