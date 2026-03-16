import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: '/NeuralEstimators.jl/dev/',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: [
{ text: 'Home', link: '/index' },
{ text: 'Methodology', link: '/methodology' },
{ text: 'Workflow overview', link: '/overview' },
{ text: 'Examples', collapsed: false, items: [
{ text: 'Replicated unstructured data', link: '/examples/data_replicated' },
{ text: 'Gridded data', link: '/examples/data_gridded' },
{ text: 'Irregular spatial data', link: '/examples/data_irregularspatial' }]
 },
{ text: 'Advanced usage', link: '/examples/advancedusage' },
{ text: 'API', collapsed: false, items: [
{ text: 'Parameters and data', link: '/API/parametersdata' },
{ text: 'Estimators', link: '/API/estimators' },
{ text: 'Training', link: '/API/training' },
{ text: 'Post-training assessment', link: '/API/assessment' },
{ text: 'Inference with observed data', link: '/API/inference' },
{ text: 'Neural-network building blocks', link: '/API/architectures' },
{ text: 'Approximate distributions', link: '/API/approximatedistributions' },
{ text: 'Loss functions', link: '/API/lossfunctions' },
{ text: 'Miscellaneous', link: '/API/miscellaneous' },
{ text: 'Index', link: '/API/index' }]
 }
]
,
}

// const nav = [
//   ...navTemp.nav,
//   {
//     component: 'VersionPicker'
//   }
// ]
const nav = [
  ...navTemp.nav
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/NeuralEstimators.jl/dev/',// TODO: replace this in makedocs!
  title: 'NeuralEstimators.jl',
  description: 'Documentation for NeuralEstimators.jl',
  lastUpdated: true,
  cleanUrls: true,
  outDir: '../1', // This is required for MarkdownVitepress to work correctly...
  head: [
    
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  
  markdown: {
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
      mathjax.markdownConfig(md);
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  vite: {
    plugins: [
      mathjax.vitePlugin,
    ],
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('/NeuralEstimators.jl'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ], 
    }, 
    ssr: { 
      noExternal: [ 
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ], 
    },
  },
  themeConfig: {
    outline: 'deep',
    logo: { src: '/logo.png', width: 24, height: 24},
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: [
{ text: 'Home', link: '/index' },
{ text: 'Methodology', link: '/methodology' },
{ text: 'Workflow overview', link: '/overview' },
{ text: 'Examples', collapsed: false, items: [
{ text: 'Replicated unstructured data', link: '/examples/data_replicated' },
{ text: 'Gridded data', link: '/examples/data_gridded' },
{ text: 'Irregular spatial data', link: '/examples/data_irregularspatial' }]
 },
{ text: 'Advanced usage', link: '/examples/advancedusage' },
{ text: 'API', collapsed: false, items: [
{ text: 'Parameters and data', link: '/API/parametersdata' },
{ text: 'Estimators', link: '/API/estimators' },
{ text: 'Training', link: '/API/training' },
{ text: 'Post-training assessment', link: '/API/assessment' },
{ text: 'Inference with observed data', link: '/API/inference' },
{ text: 'Neural-network building blocks', link: '/API/architectures' },
{ text: 'Approximate distributions', link: '/API/approximatedistributions' },
{ text: 'Loss functions', link: '/API/lossfunctions' },
{ text: 'Miscellaneous', link: '/API/miscellaneous' },
{ text: 'Index', link: '/API/index' }]
 }
]
,
    editLink: { pattern: "https://github.com/msainsburydale/NeuralEstimators.jl/edit/main/docs/src/:path" },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/msainsburydale/NeuralEstimators.jl' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
