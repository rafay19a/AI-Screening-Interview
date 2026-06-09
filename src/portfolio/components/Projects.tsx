import { motion } from 'framer-motion';
import { ExternalLink, ShoppingCart, Plane, Utensils, Hotel, Sparkles } from 'lucide-react';

const projects = [
  {
    title: "Buttertune",
    description: "Full-stack e-commerce platform for organic skincare and beauty products. Features catalog, cart, and secure checkout.",
    link: "https://buttertuneskincare.com/",
    tags: ["Full Stack", "E-commerce", "SEO"],
    icon: <ShoppingCart className="w-6 h-6" />,
    color: "from-pink-500/20 to-rose-500/20"
  },
  {
    title: "Roameo PMS",
    description: "Advanced hotel management system targeting resorts. Handles bookings, property management, and admin dashboards.",
    link: "https://roameo-pms.vercel.app/",
    tags: ["SaaS", "Management", "Real Estate"],
    icon: <Hotel className="w-6 h-6" />,
    color: "from-blue-500/20 to-indigo-500/20"
  },
  {
    title: "Kartarpur Journey",
    description: "A specialized tourism platform for Sikh heritage sites, facilitating seamless travel experiences.",
    link: "https://kartarpur-journey-2uev.vercel.app",
    tags: ["Tourism", "Next.js", "UX Design"],
    icon: <Plane className="w-6 h-6" />,
    color: "from-orange-500/20 to-amber-500/20"
  },
  {
    title: "Ora De Nuit",
    description: "Modern web presence for an American restaurant, featuring dynamic menus and booking capabilities.",
    link: "https://ora-de-nuit.vercel.app/",
    tags: ["Hospitality", "Restaurant", "UI/UX"],
    icon: <Utensils className="w-6 h-6" />,
    color: "from-red-500/20 to-orange-500/20"
  },
  {
    title: "Shama-e-Hayat",
    description: "Expansion platform for an organic candle business, transitioning from Pakistan to the French market.",
    link: "https://shama-e-hayat.vercel.app/",
    tags: ["Global", "E-commerce", "Growth"],
    icon: <Sparkles className="w-6 h-6" />,
    color: "from-yellow-500/20 to-amber-500/20"
  }
];

export const Projects = () => {
  return (
    <section id="projects" className="py-24 bg-slate-950 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-white mb-4">Selected Works</h2>
          <p className="text-slate-400 max-w-2xl">
            A collection of web applications where I've led development, focusing on performance, scalability, and user experience.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {projects.map((project, index) => (
            <motion.div
              key={project.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className={`group relative p-8 rounded-3xl border border-slate-800 bg-slate-900/40 hover:bg-slate-900/60 transition-all overflow-hidden`}
            >
              {/* Gradient background effect */}
              <div className={`absolute inset-0 bg-gradient-to-br ${project.color} opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none`} />

              <div className="relative z-10">
                <div className="mb-6 p-3 w-fit rounded-2xl bg-slate-800 text-white border border-slate-700">
                  {project.icon}
                </div>

                <h3 className="text-2xl font-bold text-white mb-3 group-hover:text-primary-foreground transition-colors flex items-center gap-2">
                  {project.title}
                  <ExternalLink className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                </h3>

                <p className="text-slate-400 mb-6 leading-relaxed">
                  {project.description}
                </p>

                <div className="flex flex-wrap gap-2 mb-8">
                  {project.tags.map(tag => (
                    <span key={tag} className="text-xs font-medium px-3 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                      {tag}
                    </span>
                  ))}
                </div>

                <a
                  href={project.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-sm font-semibold text-white hover:underline"
                >
                  Visit Project
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
