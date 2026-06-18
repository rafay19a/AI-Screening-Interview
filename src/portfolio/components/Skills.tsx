import { motion } from 'framer-motion';

const skillCategories = [
  {
    title: "AI & Intelligence",
    skills: ["Natural Language Processing", "LLM Fine-tuning", "Prompt Architecture", "Whisper STT", "Semantic Search", "OpenAI Ecosystem"]
  },
  {
    title: "Web Engineering",
    skills: ["React / Next.js", "TypeScript", "Python Systems", "Streamlit", "Node.js Architecture", "REST API Design"]
  },
  {
    title: "Data & Scaling",
    skills: ["PostgreSQL", "Supabase Infrastructure", "Machine Learning Evaluation", "Data Pipelines", "SEO Optimization", "Docker"]
  }
];

export const Skills = () => {
  return (
    <section id="capabilities" className="py-48 bg-black px-6 md:px-12 border-t border-white/5">
      <div className="max-w-screen-2xl mx-auto grid grid-cols-1 md:grid-cols-12 gap-24">
        <div className="md:col-span-4">
          <h2 className="text-5xl font-medium tracking-tighter-premium sticky top-32">Capabilities<br />& Stack.</h2>
        </div>

        <div className="md:col-span-8 space-y-32">
          {skillCategories.map((cat) => (
            <motion.div
              key={cat.title}
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
              className="border-l border-white/10 pl-12"
            >
              <h3 className="text-[10px] uppercase tracking-[0.5em] text-white/30 mb-12">{cat.title}</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6">
                {cat.skills.map(skill => (
                  <span key={skill} className="text-2xl font-light text-white/70 hover:text-white transition-colors duration-500">{skill}</span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
