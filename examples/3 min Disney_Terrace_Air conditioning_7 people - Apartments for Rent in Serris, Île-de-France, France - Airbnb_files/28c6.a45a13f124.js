__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),Object.defineProperty(e,"FadeEffect",{enumerable:!0,get:function(){return t.FadeEffect}}),Object.defineProperty(e,"RelativeMoveEffect",{enumerable:!0,get:function(){return f.RelativeMoveEffect}}),Object.defineProperty(e,"ScaleEffect",{enumerable:!0,get:function(){return n.ScaleEffect}});var t=r(d[0]),f=r(d[1]),n=r(d[2])},"06dba3",["e9cc00","c78ea4","22d731"]);
__d(function(g,r,i,a,m,e,d){"use strict";var o=r(d[0]).default;Object.defineProperty(e,"__esModule",{value:!0}),e.default=void 0;var t=r(d[1]),n=r(d[2]),l=r(d[3]),p=r(d[4]),s=o(r(d[5])),u=r(d[6]),c=r(d[7]),v=r(d[8]),y=(r(d[9]),r(d[10]));const P={installers:[],injectProps:(0,s.default)([c.HyperloopStateToken,v.LegacyTrebuchetDataToken,n.FlaggerReactClientDataToken,u.AppToken,t.ThemeIdToken],(o,t,{features:n,airParams:l},p,s,u,c)=>{const v={hyperloopProvidedValues:{...u||{},...c||{}},hyperloopContextValues:{themeId:s,...o,trebuchets:t,features:n,routes:p,airParams:l}};return()=>v},{consumerId:'injectHyperloopProvidedProps'}),hoc:function(o){function t({hyperloopProvidedValues:t,...n}){return(0,y.jsx)(p.HyperloopProvidedContext.Provider,{value:t,children:(0,y.jsx)(l.HyperloopPropsContext.Provider,{value:n.hyperloopContextValues,children:(0,y.jsx)(o,{...n})})})}return t.displayName=`WithHyperloopProvidedValues(${o?.displayName||o?.name||'Component'})`,t}};e.default=P},"160b7f",["ba7a76","3b84f8","ab7b9c","ab9511","a78e15","3dbc1a","b51a5d","53ae4d","c385a8","19b173","b8c07d"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.ScaleEffect=void 0;var t=r(d[0]),n=r(d[1]);const s='scale',f=t=>{if(!t)return'';let f='';const{startScale:o,endScale:c}=t,l=new Map;return(0,n.setIfDefined)('--view-transition-scale-effect_from',o,l),(0,n.setIfDefined)('--view-transition-scale-effect_to',c,l),(0,n.setCommonOptions)(s,t,l),l.forEach((t,n)=>{f+=`${n}: ${t};`}),f},o=t.cssFragment`
  @keyframes ${s} {
    from {
      filter: var(--view-transition_from-filter);
      visibility: visible;
      scale: var(--view-transition-scale-effect_from, 1);
    }

    to {
      filter: var(--view-transition_to-filter);
      scale: var(--view-transition-scale-effect_to, 0);
    }
  }
`,c=e.ScaleEffect={name:s,apply:n=>t.cssFragment`
      ${o}
      ${f(n)}
      animation: ${c.animation};
    `,animation:(0,n.makeAnimationDefaults)(s)}},"22d731",["4786a8","d3ae23"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.THEME_IDS=e.LUXURY_RETREATS_THEME=e.DEFAULT_THEME=e.CURRENT_THEME_KEY=e.CURRENT_STYLE_SHEET_KEY=e.CHINA_CEREAL_THEME=e.CEREAL_THEME=void 0;e.CURRENT_THEME_KEY='airbnb-dls-web withStyles theme',e.CURRENT_STYLE_SHEET_KEY='airbnb-dls-web withStyles';const E=e.CEREAL_THEME=1,_=e.CHINA_CEREAL_THEME=3,T=e.LUXURY_RETREATS_THEME=6;e.DEFAULT_THEME=E,e.THEME_IDS=[E,_,T]},"26b873",[]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.PROGRESSIVE_BREAKPOINTS_UNKNOWN=e.BREAKPOINT_NAMES=void 0,Object.defineProperty(e,"PROGRESSIVE_BREAKPOINT_NAMES",{enumerable:!0,get:function(){return A.PROGRESSIVE_BREAKPOINT_NAMES}}),e.default=e.breakpoints=void 0,e.deprecatedBreakpointFromBreakpoint=function(A){if(A===n.XLARGE)return n.LARGE;return A},e.isBelowLarge=function(E){return E.isBreakpointKnown&&!E[A.LARGE_AND_ABOVE]},e.isBelowMedium=function(E){return E.isBreakpointKnown&&!E[A.MEDIUM_AND_ABOVE]},e.isBelowXlarge=function(E){return E.isBreakpointKnown&&!E[A.XLARGE_AND_ABOVE]},e.progressiveBreakpointsFromBreakpoint=function(A){return A&&_[A]||o},e.unit=e.responsive=void 0;var A=r(d[0]);const E=e.unit=8,n=e.BREAKPOINT_NAMES={XLARGE:'xlarge',LARGE:'large',MEDIUM:'medium',SMALL:'small',XSMALL:'xsmall'},t=e.breakpoints={xlarge:1440,large:1128,medium:744,small:327,xsmall:0},o=(e.responsive={[n.LARGE]:`@media (min-width: ${t[n.LARGE]}px)`,[n.MEDIUM]:`@media (max-width: ${t[n.LARGE]-1}px)`,[n.SMALL]:`@media (max-width: ${t[n.MEDIUM]-1}px)`,[A.MEDIUM_AND_ABOVE]:`@media (min-width: ${t[n.MEDIUM]}px)`,[A.LARGE_AND_ABOVE]:`@media (min-width: ${t[n.LARGE]}px)`,[A.XLARGE_AND_ABOVE]:`@media (min-width: ${t[n.XLARGE]}px)`,print:'@media print',prefersReducedMotion:'@media (prefers-reduced-motion)'},e.PROGRESSIVE_BREAKPOINTS_UNKNOWN={isBreakpointKnown:!1,[A.MEDIUM_AND_ABOVE]:!1,[A.LARGE_AND_ABOVE]:!1,[A.XLARGE_AND_ABOVE]:!1}),_={[n.SMALL]:{isBreakpointKnown:!0,[A.MEDIUM_AND_ABOVE]:!1,[A.LARGE_AND_ABOVE]:!1,[A.XLARGE_AND_ABOVE]:!1},[n.MEDIUM]:{isBreakpointKnown:!0,[A.MEDIUM_AND_ABOVE]:!0,[A.LARGE_AND_ABOVE]:!1,[A.XLARGE_AND_ABOVE]:!1},[n.LARGE]:{isBreakpointKnown:!0,[A.MEDIUM_AND_ABOVE]:!0,[A.LARGE_AND_ABOVE]:!0,[A.XLARGE_AND_ABOVE]:!1},[n.XLARGE]:{isBreakpointKnown:!0,[A.MEDIUM_AND_ABOVE]:!0,[A.LARGE_AND_ABOVE]:!0,[A.XLARGE_AND_ABOVE]:!0}};const B={flatButton:{size:6*E,sizeSmall:4*E,iconSize:3*E,iconSizeSmall:2*E},countBadge:{height:20,minWidth:20},progressBar:{height:E},rule:{heroWidth:6*E},checkBox:{size:18,checkMarkSize:32},switch:{size:4*E,backgroundWidth:6*E},radioButton:{size:2.25*E,sizeLarge:3*E,dotSize:.75*E,dotSizeLarge:E}};e.default=B},"39778f",["da2620"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.ThemeIdToken=e.LinariaThemeStylesToken=e.LinariaThemeIdToken=e.InitialCurrentBreakpointToken=e.FormFactorToken=e.DirectionToken=e.DeprecatedFormFactorToken=e.ColorSchemeToken=void 0;var t=r(d[0]),o=r(d[1]),n=r(d[2]),T=r(d[3]);e.DirectionToken=(0,n.createToken)('DirectionToken',{hydrate:!0,getDefault:()=>'ltr'}),e.InitialCurrentBreakpointToken=(0,n.createToken)('InitialCurrentBreakpointToken',{hydrate:!0,getDefault:()=>{}}),e.DeprecatedFormFactorToken=(0,n.createToken)('DeprecatedFormFactorToken',{hydrate:!0,getDefault:()=>t.FormFactor.MOBILE}),e.FormFactorToken=(0,n.createToken)('FormFactorToken',{hydrate:!0,getDefault:()=>o.DEFAULT_FORM_FACTOR}),e.ThemeIdToken=(0,n.createToken)('ThemeIdToken',{hydrate:!0,getDefault:()=>T.DEFAULT_THEME}),e.LinariaThemeIdToken=(0,n.createToken)('LinariaThemeIdToken',{hydrate:!0,getDefault:()=>{}}),e.LinariaThemeStylesToken=(0,n.createToken)('LinariaThemeStylesToken',{hydrate:!0,getDefault:()=>{}}),e.ColorSchemeToken=(0,n.createToken)('ColorSchemeToken',{hydrate:!0,getDefault:()=>{}})},"3b84f8",["4a8cb0","b679e8","cd87be","26b873"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default=function(o,s,u){return({provide:f})=>{f((0,n.keyOf)(t.RenderDependenciesToken,u.consumerId),[c.PropServiceToken,...o],async(n,...t)=>{const c=await s(...t);n.registerSelector(n=>c(n))},{trace:!1,lazy:!1,...u})}};var n=r(d[0]),t=r(d[1]),c=r(d[2])},"3dbc1a",["cd87be","b51a5d","c05363"]);
__d(function(g,r,i,a,m,e,d){},"45b9e1",[]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.FormFactor=e.DEFAULT_FORM_FACTOR=void 0,e.getFormFactorFromBreakpoint=function(o){return o===t.BREAKPOINT_NAMES.XLARGE||o===t.BREAKPOINT_NAMES.LARGE?n.DESKTOP:n.MOBILE},e.updateFormFactorCookie=function(t){if((0,c.getCookieSync)(E).value===t)return;(0,c.setCookieSync)(E,t,{domain:(0,o.getCookieHost)(),path:'/'})};var o=r(d[0]),t=r(d[1]),c=r(d[2]);const n=e.FormFactor={DESKTOP:'DESKTOP',MOBILE:'MOBILE'};e.DEFAULT_FORM_FACTOR=n.MOBILE;const E='cfrmfctr'},"4a8cb0",["5b85ba","39778f","13babd"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default=function(...t){return t.filter(Boolean).join(' ')}},"4b14aa",[]);
__d(function(g,r,i,a,m,e,d){"use strict";var o=r(d[0]).default;Object.defineProperty(e,"__esModule",{value:!0}),e.SeoFeaturesToken=e.LegacyBootstrapDataInstanceToken=e.InstallersToSkipToken=e.HyperloopStateToken=e.HeaderPropsToken=e.FooterPropsToken=e.BootstrapDataToken=e.AuthModalPropsToken=e.AirbnbRequestToken=void 0;var t=o(r(d[1])),n=r(d[2]);e.LegacyBootstrapDataInstanceToken=(0,n.createToken)('LegacyBootstrapDataInstanceToken',{map:!0,getDefault:()=>t.default}),(0,n.createToken)('BootstrapDataMapToken',{map:!0,getDefault:()=>({})}),e.BootstrapDataToken=(0,n.createToken)('BootstrapDataToken',{hydrate:!0,getDefault:()=>({})}),(0,n.createToken)('PreventCachingToken',{getDefault:()=>({})}),e.InstallersToSkipToken=(0,n.createToken)('InstallersToSkipToken',{map:!0,getDefault:()=>({})}),e.SeoFeaturesToken=(0,n.createToken)('SeoFeaturesToken',{getDefault:()=>({})}),(0,n.createToken)('IsWebviewToken',{getDefault:()=>!1}),(0,n.createToken)('AirbnbUserCookiesToken'),e.HyperloopStateToken=(0,n.createToken)('HyperloopStateToken',{getDefault:()=>({}),hydrate:!0}),e.AuthModalPropsToken=(0,n.createToken)('AuthModalPropsToken',{getDefault:()=>{},hydrate:!0}),e.AirbnbRequestToken=(0,n.createToken)('AirbnbRequestToken'),e.FooterPropsToken=(0,n.createToken)('FooterPropsToken',{getDefault:()=>{},hydrate:!0}),e.HeaderPropsToken=(0,n.createToken)('HeaderPropsToken',{getDefault:()=>{},hydrate:!0})},"53ae4d",["ba7a76","ef2bc3","cd87be"]);
__d(function(g,r,i,a,m,e,d){"use strict";var t=r(d[0]).default;Object.defineProperty(e,"__esModule",{value:!0}),e.getBreakpointFromCookie=function(t){if(null===t)return l[1];'string'==typeof t&&(t=parseInt(t,10));return l[t]},Object.defineProperty(e,"getCookieHost",{enumerable:!0,get:function(){return n.default}}),e.getCookieValFromBreakpoint=function(t){return u[t]},e.updateCookie=function(t){const u=(0,o.getCookieSync)(c).value;if(null===u)return void(0,o.setCookieSync)(c,`${t}`,{domain:(0,n.default)(),path:'/'});parseInt(u,10)!==t&&(0,o.setCookieSync)(c,`${t}`,{domain:(0,n.default)(),path:'/'})};var o=r(d[1]),n=t(r(d[2]));const u={small:1,medium:2,large:3,xlarge:4},l={1:'small',2:'medium',3:'large',4:'xlarge'},c='cbkp'},"5b85ba",["ba7a76","13babd","f5172f"]);
__d(function(g,r,i,a,m,e,d){"use strict";var t=r(d[0]),n=r(d[1]),o=r(d[2]),l=r(d[3]),c=r(d[4]),s=r(d[5]);const y={enter:{old:n.cssFragment`
        ${t.a11y.motion} {
          ${l.FadeEffect.apply({startOpacity:'var(--contextual-grow_start-opacity, 1)',endOpacity:'var(--contextual-grow_end-opacity, 0)',delay:'var(--contextual-grow_fade-delay, 50ms)',duration:75})};
          ${l.ScaleEffect.apply({endScale:'var(--contextual-grow_scale--end, 0.94)',duration:500})};
          animation: ${l.FadeEffect.animation}, ${l.ScaleEffect.animation};
          --view-transition_from-filter: brightness(var(--scrim-animation_end-brightness, 1));
          --view-transition_to-filter: brightness(var(--scrim-animation_start-brightness, 0.6));
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `},exit:{new:n.cssFragment`
        ${t.a11y.motion} {
          ${l.ScaleEffect.apply({startScale:'var(--contextual-grow_scale--end, 0.94)',endScale:'var(--contextual-grow_scale--start, 1)',duration:500})};
          animation: ${l.ScaleEffect.animation};
          --view-transition_from-filter: brightness(var(--scrim-animation_end-brightness, 0.6));
          --view-transition_to-filter: brightness(var(--scrim-animation_start-brightness, 1));
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }

        ${(0,c.descendantElements)()} {
          --view-transition_visibility: visible;
        }
      `}},f={enter:{old:n.cssFragment`
        ${t.a11y.motion} {
          mix-blend-mode: normal;
          ${l.FadeEffect.apply({startOpacity:1,endOpacity:0,duration:75,delay:'var(--contextual-grow_fade-delay, 50ms)',timingFunction:`${n.theme.motion.linearCurve.animationTimingFunction}`})}
          animation: ${l.FadeEffect.animation};
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `,new:n.cssFragment`
        ${t.a11y.motion} {
          mix-blend-mode: normal;
          ${l.FadeEffect.apply({startOpacity:'var(--contextual-grow_content-start-opacity, 0)',duration:75,delay:'var(--contextual-grow_fade-delay, 50ms)',timingFunction:`${n.theme.motion.linearCurve.animationTimingFunction}`})}
          animation: ${l.FadeEffect.animation};
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `},exit:{old:n.cssFragment`
        ${t.a11y.motion} {
          ${l.FadeEffect.apply({startOpacity:1,endOpacity:0,duration:75,delay:'var(--contextual-grow_fade-delay, 50ms)',timingFunction:`${n.theme.motion.linearCurve.animationTimingFunction}`})}
          animation: ${l.FadeEffect.animation};
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `,new:n.cssFragment`
        ${t.a11y.motion} {
          mix-blend-mode: normal;
          ${l.FadeEffect.apply({duration:75,delay:'var(--contextual-grow_fade-delay, 50ms)',timingFunction:`${n.theme.motion.linearCurve.animationTimingFunction}`})}
          animation: ${l.FadeEffect.animation};
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `}},p={enter:{old:n.cssFragment`
        ${t.a11y.motion} {
          ${l.FadeEffect.apply({duration:75,startOpacity:1,endOpacity:0,delay:'var(--contextual-grow_fade-delay, 50ms)',timingFunction:`${n.theme.motion.linearCurve.animationTimingFunction}`})};
          ${l.ScaleEffect.apply({duration:500})};
          animation: ${l.FadeEffect.animation}, ${l.ScaleEffect.animation};
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `,new:n.cssFragment`
        ${t.a11y.motion} {
          ${l.FadeEffect.apply({startOpacity:1,duration:75,delay:'var(--contextual-grow_fade-delay, 50ms)',timingFunction:`${n.theme.motion.linearCurve.animationTimingFunction}`})}
          ${l.ScaleEffect.apply()};
          animation: ${l.FadeEffect.animation};
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `},exit:{old:n.cssFragment`
        ${t.a11y.motion} {
          ${l.FadeEffect.apply({startOpacity:1,duration:75,delay:'var(--contextual-grow_fade-delay, 50ms)',timingFunction:`${n.theme.motion.linearCurve.animationTimingFunction}`})};
          animation: ${l.FadeEffect.animation};
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `,new:n.cssFragment`
        ${t.a11y.motion} {
          ${l.FadeEffect.apply({duration:75,delay:'var(--contextual-grow_fade-delay, 50ms)',timingFunction:`${n.theme.motion.linearCurve.animationTimingFunction}`})}
          animation: ${l.FadeEffect.animation};
        }

        ${t.a11y.noMotion} {
          ${l.FadeEffect.apply()};
        }
      `}},$=n.cssFragment`
  ::view-transition-group(container),
  ::view-transition-group(container-content) {
    mix-blend-mode: normal;
    overflow: hidden;
    border-radius: var(--container-border-radius, 15px);
  }

  ::view-transition-group(container) {
    border-bottom: var(--container-border, none);
  }

  &.exit::view-transition-old(container) {
    ${p.exit.old}
  }

  &.enter::view-transition-old(container) {
    ${p.enter.old};
  }

  &.enter::view-transition-new(container) {
    ${p.enter.new}
  }

  ::view-transition-new(container-content) {
    mix-blend-mode: normal;
  }

  &.enter::view-transition-old(container-content) {
    ${f.enter.old}
  }

  &.enter::view-transition-new(container-content) {
    ${f.enter.new}
  }

  &.exit::view-transition-old(container-content) {
    ${f.exit.old}
  }

  &.exit::view-transition-new(container-content) {
    ${f.exit.new}
  }

  &.exit::view-transition-new(container) {
    ${p.exit.new}
  }

  &.enter::view-transition-new(root) {
    display: none;
  }

  &.exit::view-transition-old(root) {
    display: none;
  }

  /* Scale out the old screen */
  &.enter::view-transition-old(root) {
    ${y.enter.old}
  }

  &.exit::view-transition-new(root) {
    ${y.exit.new}
  }

  /**
   * This shows all of the old elements during the transition that would
   * otherwise be hidden due to nested elements. It also ensures that if a list
   * pattern is implemented, that the singluar list item is hidden as it will be
   * promoted.
   */
  ::view-transition-group(root) {
    ::view-transition-new {
      > [data-static-element-wrapper] > [view-transition-element] {
        visibility: visible;
      }
    }

    /* stylelint-disable-next-line selector-max-type */
    ${(0,c.frozenViewTransition)('old')},
    ::view-transition-old {
      /* stylelint-disable-next-line selector-max-type */
      ${(0,c.descendantElements)()} {
        --view-transition_visibility: visible;
        /* stylelint-disable-next-line selector-max-type */
        [active-element='true'] {
          visibility: hidden;
        }
      }
    }
  }
`,u={name:o.DLSTransitionPattern.ContextualGrow,customize:t=>(0,s.extendPattern)(u,t),fragment:$,mapping:[['--contextual-grow_scale--end',{key:'endScale'}],['--contextual-grow_scale--start',{key:'startScale'}],['--contextual-grow_start-opacity',{key:'startOpacity'}],['--contextual-grow_end-opacity',{key:'endOpacity'}],['--contextual-grow_fade-delay',{key:'fadeDelay'}],['--contextual-grow_content-start-opacity',{key:'contentStartOpacity'}]]}},"5bf8dc",["daa5d1","4786a8","83da1f","06dba3","dbb634","f3229a"]);
__d(function(g,r,i,a,m,e,d){"use strict";var n=r(d[0]),t=r(d[1]),o=r(d[2]),f=r(d[3]),s=r(d[4]);const l={enter:{old:t.cssFragment`
        ${n.a11y.motion} {
          ${f.RelativeMoveEffect.apply({startX:0,endX:'calc(-1 * var(--slide-in-and-fade_offset-x-to, 200px))',duration:500})};
          ${f.FadeEffect.apply({startOpacity:1,endOpacity:0,duration:75})};
          animation: ${f.FadeEffect.animation}, ${f.RelativeMoveEffect.animation};
          animation-duration: var(--slide-in-and-fade_fade-out-duration, 75ms),
            var(--slide-in-and-fade_move-duration, 500ms);
        }

        ${n.a11y.noMotion} {
          ${f.FadeEffect.apply({startOpacity:1,endOpacity:0})};
        }
      `,new:t.cssFragment`
        ${n.a11y.motion} {
          ${f.RelativeMoveEffect.apply({endX:0,startX:'var(--slide-in-and-fade_offset-x-from, 200px)',duration:500})};
          ${f.RelativeMoveEffect.align()}
          ${f.FadeEffect.apply({startOpacity:0,endOpacity:1,delay:50,duration:350})};
          animation: ${f.FadeEffect.animation}, ${f.RelativeMoveEffect.animation};
          animation-duration: var(--slide-in-and-fade_fade-in-duration, 350ms),
            var(--slide-in-and-fade_move-duration, 500ms);
        }

        ${n.a11y.noMotion} {
          ${f.FadeEffect.apply({startOpacity:0,endOpacity:1})};
        }

        background: var(--view-transition_panel-background, #fff);
      `},exit:{old:t.cssFragment`
        ${n.a11y.motion} {
          ${f.RelativeMoveEffect.apply({startX:0,endX:'var(--slide-in-and-fade_offset-x-to, 200px)',duration:500})};
          ${f.FadeEffect.apply({startOpacity:1,endOpacity:0,duration:75})};
          animation: ${f.FadeEffect.animation}, ${f.RelativeMoveEffect.animation};
          animation-duration: var(--slide-in-and-fade_fade-out-duration, 75ms),
            var(--slide-in-and-fade_move-duration, 500ms);
        }

        ${n.a11y.noMotion} {
          ${f.FadeEffect.apply({startOpacity:1,endOpacity:0})};
        }

        background: var(--view-transition_panel-background, #fff);
      `,new:t.cssFragment`
        ${n.a11y.motion} {
          ${f.RelativeMoveEffect.apply({endX:0,startX:'calc(-1 * var(--slide-in-and-fade_offset-x-from, 200px))',duration:500})};
          ${f.FadeEffect.apply({startOpacity:0,endOpacity:1,delay:50,duration:350})};
          animation: ${f.FadeEffect.animation}, ${f.RelativeMoveEffect.animation};
          animation-duration: var(--slide-in-and-fade_fade-in-duration, 350ms),
            var(--slide-in-and-fade_move-duration, 500ms);
        }

        ${n.a11y.noMotion} {
          ${f.FadeEffect.apply({startOpacity:0,endOpacity:1})};
        }

        --view-transition_mix-blend-mode: normal;
      `}},c=t.cssFragment`
  ::view-transition-new(root) {
    animation: none;
    mix-blend-mode: normal;
  }

  ::view-transition-group(root) {
    animation: none;
  }

  ::view-transition-old(root) {
    display: none;
  }

  ::view-transition-group(screen) {
    animation: none;
  }

  ::view-transition-image-pair(screen) {
    mix-blend-mode: normal;
    display: grid;
    height: 100%;
    overflow: hidden;
  }

  ::view-transition-old(screen),
  ::view-transition-new(screen) {
    display: flex;
    grid-area: 1 / 1;
    mix-blend-mode: normal;
    position: static;
  }

  ::view-transition-new(screen) {
    animation: none;
  }

  &.enter::view-transition-old(screen) {
    ${l.enter.old}
  }

  &.enter::view-transition-new(screen) {
    ${l.enter.new}
  }

  &.exit::view-transition-new(screen) {
    ${l.exit.new}
  }

  &.exit::view-transition-old(screen) {
    ${l.exit.old}
  }
`,p={name:o.DLSTransitionPattern.SlideInAndFade,customize:n=>(0,s.extendPattern)(p,n),fragment:c,mapping:[['--slide-in-and-fade_fade-in-duration',{key:'fadeInDuration',type:'duration'}],['--slide-in-and-fade_fade-out-duration',{key:'fadeOutDuration',type:'duration'}],['--slide-in-and-fade_move-duration',{key:'moveDuration',type:'duration'}]]}},"74aca7",["daa5d1","4786a8","83da1f","06dba3","f3229a"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default=function({consume:u}){u([n.LegacyTrebuchetDataToken],async n=>{(0,t.setTrebuchetData)(n)},{consumerId:'installBugsnagTrebuchetConfig'})};var t=r(d[0]),n=r(d[1])},"7fafb7",["33392f","c385a8"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.IsAirliteToken=void 0;var t=r(d[0]);e.IsAirliteToken=(0,t.createToken)('IsAirliteToken',{hydrate:!0,getDefault:()=>!1})},"8903b7",["cd87be"]);
__d(function(g,r,i,a,m,e,d){"use strict";var t=r(d[0]),o=r(d[1]),n=r(d[2]),s=r(d[3]),f=r(d[4]);const l={enter:{old:o.cssFragment`
        ${t.a11y.motion} {
          ${s.RelativeMoveEffect.apply({endX:0,startX:0,endY:'calc(-1 * var(--slide-up-from-bottom_offset-y-to, 5.5%))',startScale:'var(--slide-up-from-bottom_scale--start, 1)',endScale:'var(--slide-up-from-bottom_scale--end, 0.94)',duration:500})}
          /* animation config declaration for safari support */
          animation-duration: var(--slide-up-from-bottom_slide-down-duration, 500ms);
          --view-transition_from-filter: brightness(var(--scrim-animation_start-brightness, 1));
          --view-transition_to-filter: brightness(var(--scrim-animation_end-brightness, 0.6));
        }

        ${t.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `,new:o.cssFragment`
        ${t.a11y.motion} {
          ${s.RelativeMoveEffect.apply({startY:'var(--slide-up-from-bottom_offset-y-from, 100%)',endY:'var(--slide-up-from-bottom_offset-y-to-polyfill, 0px)',timingFunction:`${o.theme.motion.enterCurve.animationTimingFunction}`,duration:500})};
          /* animation config declaration for safari support */
          animation-duration: var(--slide-up-from-bottom_slide-up-duration, 500ms);
          animation-timing-function: var(
            --view-transition_timing-function,
            ${o.theme.motion.standardCurve.animationTimingFunction}
          );
        }

        ${t.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `,oldRoot:o.cssFragment`
        ${t.a11y.motion} {
          ${s.RelativeMoveEffect.apply({startY:'var(--slide-up-from-bottom_root-offset-y-from, 24px)',endY:'var(--slide-up-from-bottom_root-offset-y-to, 2%)',startScale:'var(--slide-up-from-bottom_scale--start, 1)',endScale:'var(--slide-up-from-bottom_scale--end, 0.94)',duration:500})};
          /* animation config declaration for safari support */
          animation-duration: var(--slide-up-from-bottom_slide-down-duration, 500ms);
        }

        ${t.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `,newRoot:o.cssFragment`
        ${t.a11y.motion} {
          ${s.RelativeMoveEffect.apply({endY:'var(--slide-up-from-bottom_root-offset-y-to, 2%)',startScale:'var(--slide-up-from-bottom_scale--end, 0.94)',endScale:'var(--slide-up-from-bottom_scale--start, 1)',duration:500})};
          /* animation config declaration for safari support */
          animation-duration: var(--slide-up-from-bottom_slide-up-duration, 500ms);
        }

        ${t.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `},exit:{old:o.cssFragment`
        ${t.a11y.motion} {
          ${s.RelativeMoveEffect.apply({startY:'var(--slide-up-from-bottom_offset-y-to-polyfill, 0px)',endY:'var(--slide-up-from-bottom_offset-y-from, 100%)',timingFunction:` ${o.theme.motion.exitCurve.animationTimingFunction}`,duration:300})};
          /* animation config declaration for safari support */
          animation-duration: var(--slide-up-from-bottom_slide-down-duration, 300ms);
          animation-timing-function: var(
            --view-transition_timing-function,
            ${o.theme.motion.exitCurve.animationTimingFunction}
          );
        }

        ${t.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }

        z-index: 1;
      `,new:o.cssFragment`
        ${t.a11y.motion} {
          ${s.RelativeMoveEffect.apply({endX:0,startX:0,startY:'calc(-1 * var(--slide-up-from-bottom_offset-y-to, 5.5%))',startScale:'var(--slide-up-from-bottom_scale--end, 0.94)',endScale:'var(--slide-up-from-bottom_scale--start, 1)',duration:500})};
          /* animation config declaration for safari support */
          animation-duration: var(--slide-up-from-bottom_slide-up-duration, 500ms);
          --view-transition_from-filter: brightness(var(--scrim-animation_end-brightness, 0.6));
          --view-transition_to-filter: brightness(var(--scrim-animation_start-brightness, 1));
        }

        ${t.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `,oldRoot:o.cssFragment`
        ${t.a11y.motion} {
          ${s.RelativeMoveEffect.apply({endY:'var(--slide-up-from-bottom_root-offset-y-from, 24px)',duration:500})};
          /* animation config declaration for safari support */
          animation-duration: var(--slide-up-from-bottom_slide-down-duration, 500ms);
          --view-transition_from-filter: brightness(var(--scrim-animation_end-brightness, 0.6));
          --view-transition_to-filter: brightness(var(--scrim-animation_start-brightness, 1));
        }

        ${t.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `}},p=o.cssFragment`
  /* normal blending that allows new view to
    sit on top & hide old view */
  --view-transition_mix-blend-mode: normal;

  ::view-transition-group(screen) {
    z-index: 1;
    /* depends on gap between previous/new context sheets */
    clip-path: inset(calc(-1 * var(--slide-up-from-bottom_root-offset-y-from, 24px)) 0 0 0);
  }

  ::view-transition-group(root) {
    animation: none;
  }

  ::view-transition-old(root),
  ::view-transition-new(root) {
    display: none;
  }

  ::view-transition-old(minimized-screen),
  ::view-transition-new(minimized-screen) {
    display: none;
  }

  ::view-transition-group(minimized-screen) {
    z-index: -1;
  }

  ::view-transition-old(*),
  ::view-transition-new(*) {
    mix-blend-mode: normal;
  }

  &.enter::view-transition-old(screen) {
    ${l.enter.old};
  }

  &.enter::view-transition-new(screen) {
    ${l.enter.new};
  }

  &.exit::view-transition-old(screen) {
    ${l.exit.old};
  }

  &.exit::view-transition-new(screen) {
    ${l.exit.new};
  }

  &.enter::view-transition-old(root) {
    ${l.enter.oldRoot};
  }

  &.enter::view-transition-new(root) {
    ${l.enter.newRoot};
  }

  &.exit::view-transition-old(root) {
    ${l.exit.oldRoot};
  }
`,c={name:n.DLSTransitionPattern.SlideUpFromBottom,customize:t=>(0,f.extendPattern)(c,t),fragment:p,mapping:[['--slide-up-from-bottom_scale--end',{key:'screenEndScale'}],['--slide-up-from-bottom_scale--start',{key:'screenStartScale'}],['--slide-up-from-bottom_root-offset-y-from',{key:'rootStartY'}],['--slide-up-from-bottom_root-offset-y-to',{key:'rootEndY'}],['--slide-up-from-bottom_root-offset-x-from',{key:'rootStartX'}],['--slide-up-from-bottom_root-offset-x-to',{key:'rootEndX'}],['--slide-up-from-bottom_slide-up-duration',{key:'slideUpDuration',type:'duration'}],['--slide-up-from-bottom_slide-down-duration',{key:'slideDownDuration',type:'duration'}]]}},"929c43",["daa5d1","4786a8","83da1f","06dba3","f3229a"]);
__d(function(g,r,i,a,m,e,d){"use strict";var n=r(d[0]),t=r(d[1]),o=r(d[2]),s=r(d[3]),l=r(d[4]);const f={enter:{old:t.cssFragment`
        ${n.a11y.motion} {
          ${s.RelativeMoveEffect.apply({duration:500,endX:'-30%'})};
          animation-duration: var(--slide-in-from-edge_slide-out-duration, 500ms);
          --view-transition_from-filter: brightness(var(--scrim-animation_start-brightness, 1));
          --view-transition_to-filter: brightness(var(--scrim-animation_end-brightness, 0.6));
        }

        ${n.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `,new:t.cssFragment`
        height: 100%;

        ${n.a11y.motion} {
          ${s.RelativeMoveEffect.apply({duration:500,startX:'100%'})};
          ${s.RelativeMoveEffect.align()}
          animation-duration: var(--slide-in-from-edge_slide-in-duration, 500ms);
        }

        ${n.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `},exit:{old:t.cssFragment`
        ${n.a11y.motion} {
          ${s.RelativeMoveEffect.apply({duration:500,endX:'100%'})};
          animation-duration: var(--slide-in-from-edge_slide-out-duration, 500ms);
          height: 100%;
        }

        ${n.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }

        z-index: 1;
      `,new:t.cssFragment`
        ${n.a11y.motion} {
          ${s.RelativeMoveEffect.apply({duration:500,startX:'-30%'})};
          animation-duration: var(--slide-in-from-edge_slide-in-duration, 500ms);
          --view-transition_from-filter: brightness(var(--scrim-animation_end-brightness, 0.6));
          --view-transition_to-filter: brightness(var(--scrim-animation_start-brightness, 1));
        }

        ${n.a11y.noMotion} {
          ${s.FadeEffect.apply()};
        }
      `}},v=t.cssFragment`
  ::view-transition-new(root) {
    animation: none;
    opacity: 1;
    mix-blend-mode: normal;
  }

  ::view-transition-group(root) {
    animation: none;
  }

  ::view-transition-old(root) {
    display: none;
  }

  ::view-transition-group(screen) {
    clip-path: inset(0 0 -400px 0);
    animation: none;
  }

  ::view-transition-image-pair(screen) {
    display: grid;
    height: 100%;
  }

  ::view-transition-new(screen),
  ::view-transition-old(screen) {
    background: var(--view-transition_panel-background, #fff);
    grid-area: 1 / 1;
    position: static;
    mix-blend-mode: normal;
    display: flex;
  }

  &.enter::view-transition-new(screen) {
    ${f.enter.new}
  }

  &.enter::view-transition-old(screen) {
    ${f.enter.old}
  }

  &.exit::view-transition-new(screen) {
    ${f.exit.new}
  }

  &.exit::view-transition-old(screen) {
    ${f.exit.old}
  }
`,c={name:o.DLSTransitionPattern.SlideInFromEdge,customize:n=>(0,l.extendPattern)(c,n),fragment:v,mapping:[['--slide-in-from-edge_slide-in-duration',{key:'slideInDuration',type:'duration'}],['--slide-in-from-edge_slide-out-duration',{key:'slideOutDuration',type:'duration'}]]}},"958172",["daa5d1","4786a8","83da1f","06dba3","f3229a"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.HyperloopProvidedContext=void 0;var o=r(d[0]);r(d[1]);e.HyperloopProvidedContext=(0,o.createContext)({})},"a78e15",["07aa1f","b8c07d"]);
__d(function(g,r,i,a,m,e,d){"use strict";function t(){const n=r(d[0]);return t=function(){return n},n}Object.defineProperty(e,"__esModule",{value:!0}),e.default=function(o){const{polyfill:c,ready:u}=l.MagicTransitionState,{style:y,className:w}=l.MagicTransitionState.instance,f=document.documentElement.classList,{viewTransition_native:E,viewTransition_slideInAndFade:_,viewTransition_slideInFromEdge:h,viewTransition_slideUpFromBottom:M,viewTransition_contextualGrow:P}=v;f.toggle('dir',!0),f.toggle('native',!0),p(E),p(_),p(h),p(M),p(P),(0,t().effect)(()=>{const t=Array.from(w.value||[]),n=s.motionPreference.user.value,o=s.motionPreference.system.value;o&&'no-motion'===n&&document.documentElement.style.setProperty('--reduced-motion_duration','150ms'),o&&'no-animation'===n&&document.documentElement.style.setProperty('--reduced-motion_duration','0.00000000001s'),t.forEach(t=>{t.split(' ').forEach(t=>f.toggle(t,!0))});const l={...y.value};return Object.keys(l).forEach(t=>{document.documentElement.style.setProperty(t,`${l[t]}`)}),()=>{t.forEach(t=>{f.remove(...t.split(' '))}),Object.keys(l).forEach(t=>{document.documentElement.style.removeProperty(t)}),document.documentElement.style.removeProperty('--reduced-motion_duration')}}),null===s.motionPreference.system.peek()&&(0,s.initialize)(o);l.startReactTransition.peek()!==n.startTransition&&(l.startReactTransition.value=n.startTransition);l.ScrollDriven.ready.value||l.ScrollDriven.install();if(!c.filled.peek()){const t='startViewTransition'in document&&null!==document.startViewTransition;t&&(c.startViewTransition.value=document.startViewTransition.bind(document)),document.startViewTransition=T,c.supportsNative.value=t,c.filled.value=!0,l.MagicTransitionState.native.value=t,u.peek()||requestAnimationFrame(()=>{u.value=!0})}};r(d[1]);var n=r(d[2]),o=r(d[3]),s=r(d[4]),l=r(d[5]),c=r(d[6]),u=r(d[7]);r(d[8]),r(d[9]),r(d[10]),r(d[11]);const v={viewTransition_native:"vz2oe5x",viewTransition:"vg7vsjx",viewTransitionContainer:"v1cv8r21",viewTransition_contextualGrow:"vyb6402",viewTransition_slideInAndFade:"v1koiow6",viewTransition_slideInFromEdge:"vrbhsjc",viewTransition_slideUpFromBottom:"vgue9iu"},p=t=>t.split(' ').forEach(t=>document.documentElement.classList.toggle(t,!0));async function y(){const{instance:n}=l.MagicTransitionState;await new Promise(t=>requestAnimationFrame(()=>setTimeout(t,0))),(0,t().batch)(()=>{n.transitionGroups.value=new Map,l.MagicTransitionState.anyActive.value=!1,l.MagicTransitionState.middleware.forEach(t=>t.onTransitionEnd?.()),n.active.value=!1,n.className.value=null,n.style.value=null,requestAnimationFrame(()=>{document.documentElement.style.setProperty('--view-transition_capture-old',null),document.documentElement.style.setProperty('--view-transition_capture-new',null)})})}function w(){const{instance:n}=l.MagicTransitionState;(0,t().batch)(()=>{document.documentElement.style.setProperty('--view-transition_capture-old',' '),document.documentElement.style.setProperty('--view-transition_capture-new','initial'),l.MagicTransitionState.anyActive.value=!0,l.MagicTransitionState.middleware.forEach(t=>t.onTransitionStart?.()),n.layers.value=0,n.active.value=!0,n.transitionGroups.value=new Map})}const T=t=>{const{activeViewTransition:n,polyfill:s,instance:v}=l.MagicTransitionState,p='object'==typeof t,T=p?t.update:t;if(p&&s.supportsNative.peek()){const{classNames:n,enableHistory:o,style:s}=t;(0,c.configure)({className:n,enableHistory:o,style:s})}if(s.supportsNative.peek()){w();const c=s.startViewTransition;if(c.value){const s=async()=>{let t;return(0,o.flushSync)(()=>{t=T?.()}),await t,l.MagicTransitionState.middleware.forEach(t=>{t.onTransitionDOMUpdated&&v.domUpdateCallbacks.add(t.onTransitionDOMUpdated)}),await v.wait(),document.documentElement.style.setProperty('--view-transition_capture-old','initial'),document.documentElement.style.setProperty('--view-transition_capture-new',' '),t};let p;const w=t?.types;if(w)try{p=c.value({types:w,update:s})}catch{s(),p=(0,u.getFakeTransition)()}else p=c.value(s);return p.finished.then(async()=>{y(),n.value=null}),n.value=p,p}}if(!s.supportsNative.peek()){const t=(0,u.getFakeTransition)();return T?.(),t}return{}}},"a8ceeb",["a954a0","ea4b89","07aa1f","b67917","daa5d1","83da1f","c32f72","f32303","5bf8dc","74aca7","958172","929c43"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.HyperloopPropsContext=void 0;var o=r(d[0]);e.HyperloopPropsContext=(0,o.createContext)({})},"ab9511",["07aa1f"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default=function({consume:o}){o([n.EnvToken],({ENV_ROLE:n,IS_DEV:o})=>{(0,t.setState)({HYPERLOOP_ENV:n||'',IS_DEV:o})},{consumerId:'installHyperloopState'})};var t=r(d[0]),n=r(d[1])},"b0591a",["46951f","b51a5d"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.FORM_FACTOR_COOKIE=e.FORM_FACTOR_BREAKPOINT=e.FORM_FACTOR=e.DEFAULT_FORM_FACTOR=void 0;const O=e.FORM_FACTOR={COMPACT:'compact',WIDE:'wide'};e.FORM_FACTOR_BREAKPOINT='mediumAndAbove',e.DEFAULT_FORM_FACTOR=O.COMPACT,e.FORM_FACTOR_COOKIE='frmfctr'},"b679e8",[]);
__d(function(g,r,i,a,m,e,d){"use strict";var t=r(d[0]).default;Object.defineProperty(e,"__esModule",{value:!0}),e.default=function({consume:t}){t([n.IsAirliteToken],t=>{(0,u.default)(t)},{consumerId:'ViewTransition'})};var n=r(d[1]),u=t(r(d[2]))},"bf022b",["ba7a76","8903b7","a8ceeb"]);
__d(function(g,_r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.wrapWithKrakenRoutingConfig=function n(r,o){if('true'!==process.env.CI&&'node'!==process?.release?.name)return r;r.krakenRoutingConfig||(r.krakenRoutingConfig=o);r.routes&&(r.routes=r.routes?.map(r=>n(r,[])));return r}},"c0f1c1",[]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),Object.defineProperty(e,"configure",{enumerable:!0,get:function(){return n.configure}});var n=r(d[0])},"c32f72",["5e4d5c"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.LegacyTrebuchetToken=e.LegacyTrebuchetDataToken=void 0;var t=r(d[0]);(0,t.createToken)('TrebuchetToken'),e.LegacyTrebuchetDataToken=(0,t.createToken)('LegacyTrebuchetDataToken',{getDefault:()=>({})}),e.LegacyTrebuchetToken=(0,t.createToken)('LegacyTrebuchetToken')},"c385a8",["cd87be"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.RelativeMoveEffect=void 0;var t=r(d[0]),n=r(d[1]);const o='relative-move',v=t=>{if(!t)return'';let v='';const{startX:f,startY:s,endX:l,endY:c,startScale:w,endScale:_}=t,p=new Map;return(0,n.setIfDefined)('--view-transition-relative-move-effect_from-x',f,p),(0,n.setIfDefined)('--view-transition-relative-move-effect_from-y',s,p),(0,n.setIfDefined)('--view-transition-relative-move-effect_to-x',l,p),(0,n.setIfDefined)('--view-transition-relative-move-effect_to-y',c,p),(0,n.setIfDefined)('--view-transition-relative-move-effect_scale-from',w,p),(0,n.setIfDefined)('--view-transition-relative-move-effect_scale-to',_,p),(0,n.setCommonOptions)(o,t,p),p.forEach((t,n)=>{v+=`${n}: ${t};`}),v},f=t.cssFragment`
  /* normal blending to allow new view to
    sit on top & hide old view */
  --view-transition_mix-blend-mode: normal;
  --view-transition-fade-effect_opacity-from: 1;

  @keyframes ${o} {
    from {
      filter: var(--view-transition_from-filter);
      visibility: visible;
      transform: translate(
        var(--view-transition-relative-move-effect_from-x, 0),
        var(--view-transition-relative-move-effect_from-y, 0)
      );
      scale: var(--view-transition-relative-move-effect_scale-from, 1);
    }

    to {
      filter: var(--view-transition_to-filter);
      transform: translate(
        var(--view-transition-relative-move-effect_to-x, 0),
        var(--view-transition-relative-move-effect_to-y, 0)
      );
      scale: var(--view-transition-relative-move-effect_scale-to, 1);
    }
  }
`,s=e.RelativeMoveEffect={name:o,align:()=>"\n    margin-top: calc(\n      var(--view-transition-group-new_top, 0) - var(--view-transition-group-old_top, 0)\n    );\n  ",apply:n=>t.cssFragment`
      ${f}
      ${v(n)}
      animation: ${s.animation};
    `,animation:(0,n.makeAnimationDefaults)(o)}},"c78ea4",["4786a8","d3ae23"]);
__d(function(g,r,i,a,m,e,d){"use strict";var t=r(d[0]).default;function n(){const u=t(r(d[1]));return n=function(){return u},u}Object.defineProperty(e,"__esModule",{value:!0}),e.default=function(t){(0,n().default)(t)}},"cdaa6a",["ba7a76","bf022b"]);
__d(function(g,r,i,_a,m,e,d){"use strict";var t=r(d[0]).default;Object.defineProperty(e,"__esModule",{value:!0}),e.benchmarkLogging=function({api_used:t,version:o,method:u,loggingKeyArgIndex:c}){let a=null,f={};const p=s({method:u}),y=({key:u,time:c})=>{p({key:u,time:c,counterMap:f}),a&&clearTimeout(a),0!==Object.keys(f).length&&(a=setTimeout(()=>{l({api_used:t,version:o,counterMap:f}),f={}},n.LOGGING_DEBOUNCE_MS))};return t=>(...o)=>{if(100*Math.random()<n.SAMPLE_RATE)return t(...o);const u=performance.now(),s=t(...o),a=_(o,c);return y({key:a,time:Math.floor(performance.now()-u)}),s}};var n=r(d[1]),o=t(r(d[2])),u=t(r(d[3]));const c=(...t)=>t.join(':');function s({method:t}){return window.navigator&&window.navigator.sendBeacon?({key:n,time:o,counterMap:u})=>{const s=c(t,n);s in u||(u[s]={counter:0,times:[]}),u[s].counter+=1,u[s].times.push(o)}:()=>{}}const a=({counter:t,times:n})=>{if(0===t)return 0;const o=n.reduce((t,n)=>t+n,0);return Math.floor(o/t)},f=t=>t.split(':'),p=t=>{const[n,o]=f(t);return[`method:${n}`,`key:${(0,u.default)(o)}`]};function l({api_used:t,version:n,counterMap:u}){const c=Object.keys(u).map(o=>{const{counter:c,times:s}=u[o],f=p(o);return{type:'distribution',metric:'frontend.cookie.operation_time',value:a({counter:c,times:s}),tags:[...f,`api_used:${t}`,`version:${n}`]}});(0,o.default)(c)}function _(t,n){const o=t[n||0];return'string'==typeof o?o:'no_key'}},"d0e903",["ba7a76","973ca3","130c13","65cdce"]);
__d(function(g,r,i,a,m,e,d){"use strict";function n(n){return`--view-transition-${n}-effect`}function t(t,o,u){const s=n(t);return void 0!==u?`var(${s}_${o}, ${u})`:`var(${s}_${o})`}Object.defineProperty(e,"__esModule",{value:!0}),e.makeAnimationDefaults=function(n,u){const{delay:s,duration:c,timingFunction:$}={...o,...u},f=`var(--reduced-motion_duration, ${t(n,'duration',`${c}ms`)})`,v=t(n,'timing-function',$),_=t(n,'delay',s);return`${f} ${v} ${_} 1 normal var(--view-transition_fill-mode, both) var(--view-transition_play-state,paused) ${n}`},e.setCommonOptions=function(t,o,s){const c=n(t),{duration:$,delay:f,timingFunction:v}=o,_='number'==typeof f?`${f}ms`:f;u(`${c}_duration`,$?`${$}ms`:void 0,s),u(`${c}_transition_timing-function`,v,s),u(`${c}_delay`,_,s)},e.setIfDefined=u;const o={delay:'0ms',duration:200,timingFunction:r(d[0]).theme.motion.standardCurve.animationTimingFunction};function u(n,t,o){void 0!==t&&o.set(n,t)}},"d3ae23",["4786a8"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.deleteCookieAsync=async function(t,s){if(n())return window.cookieStore.delete(t,s);return new Promise(n=>{(0,o.setCookie)(t,'',{...s,expires:-1}),n()})},e.getAllCookiesAsync=async function(){if(n())return window.cookieStore.getAll();return await Promise.all(Array.from(document.cookie.split(/;\s*/)).map(async o=>{const[n,t]=o.split('=');return{name:n,value:decodeURIComponent(t)}}))},e.getCookieAsync=async function(t,s){if(!t)return null;if(n())return window.cookieStore.get(t,s);return Promise.resolve({name:t,value:(0,o.getCookie)(t)})},e.setCookieAsync=async function(t,s,c){if(n())return window.cookieStore.set(t,s,c);return new Promise(n=>{(0,o.setCookie)(t,s,c),n()})};var o=r(d[0]);function n(){return'cookieStore'in window}},"d518f3",["e90d84"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.XLARGE_AND_ABOVE=e.PROGRESSIVE_BREAKPOINT_NAMES=e.MEDIUM_AND_ABOVE=e.LARGE_AND_ABOVE=void 0;const A=e.PROGRESSIVE_BREAKPOINT_NAMES={MEDIUM_AND_ABOVE:'mediumAndAbove',LARGE_AND_ABOVE:'largeAndAbove',XLARGE_AND_ABOVE:'xlargeAndAbove'},{MEDIUM_AND_ABOVE:E,LARGE_AND_ABOVE:_,XLARGE_AND_ABOVE:O}=A;e.XLARGE_AND_ABOVE=O,e.LARGE_AND_ABOVE=_,e.MEDIUM_AND_ABOVE=E},"da2620",[]);
__d(function(g,r,i,a,m,_e,d){"use strict";function e(){const n=r(d[0]);return e=function(){return n},n}Object.defineProperty(_e,"__esModule",{value:!0}),_e.a11y=void 0,_e.initialize=function(e){const t=navigator.userAgent.includes('OS X'),o=navigator.userAgent.includes('iPhone')||navigator.userAgent.includes('iPad');if(!('matchMedia'in window))return void(n.system.value=!1);let s;try{s=matchMedia('(prefers-reduced-motion: reduce)'),n.system.value=s.matches||!!e,window.motionPreference=n}catch{return void(n.system.value=!1)}function u(e){'fast'===n.update.peek()?(n.system.value=e,n.user.value=e?t||o?'no-motion':'no-animation':'motion'):n.user.value='no-animation'}const c=matchMedia('(update: slow)').matches,l=matchMedia('(update: none)').matches;(c||l)&&(n.update.value=c?'slow':'none');u(n.system.value),s.addEventListener?.('change',e=>{u(e.matches)})},_e.motionPreference=void 0;const n=_e.motionPreference={system:(0,e().signal)(null),user:(0,e().signal)('motion'),update:(0,e().signal)('fast')};_e.a11y={motion:'@media (prefers-reduced-motion: no-preference)',noMotion:'@media (prefers-reduced-motion: reduce), (update: slow), (update: none)'}},"daa5d1",["a954a0"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.descendantElements=function(){return'[data-static-element-wrapper] > [view-transition-element] [view-transition-element]'},e.frozenViewTransition=function(n){return`[frozen-view-transition-${n}]`}},"dbb634",[]);
__d(function(g,r,i,a,m,e,d){"use strict";var o=r(d[0]).default;Object.defineProperty(e,"__esModule",{value:!0}),e.getCookie=e.default=void 0;var t=r(d[1]),n=o(r(d[2])),s=r(d[3]);const c=(0,s.benchmarkLogging)({api_used:'document',method:'default',version:'1.0.0'})(n.default);e.default=c;(0,s.benchmarkLogging)({api_used:'document',method:'set',version:'1.0.0'})(n.setCookie),e.getCookie=(0,s.benchmarkLogging)({api_used:'document',method:'get',version:'1.0.0'})(n.getCookie),(0,s.benchmarkLogging)({api_used:'cookie_store',method:'set',version:'1.0.0'})(t.setCookieAsync),(0,s.benchmarkLogging)({api_used:'cookie_store',method:'get',version:'1.0.0'})(t.getCookieAsync),(0,s.benchmarkLogging)({api_used:'cookie_store',method:'delete',version:'1.0.0'})(t.deleteCookieAsync),(0,s.benchmarkLogging)({api_used:'cookie_store',method:'getAll',version:'1.0.0'})(t.getAllCookiesAsync)},"e7272f",["45f788","d518f3","e90d84","d0e903"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default=function(t,n,o){if(arguments.length>1&&(null==n||'[object Object]'!==Object.prototype.toString.call(n)))return u(t,n,o);return c(t)},e.getCookie=c,e.setCookie=u;const t=new Map;let n=null;function o(){if(t.size>0)return t;const o=g.document.cookie;return o&&o.split('; ').forEach(n=>{const[o,c]=n.split('=').map(decodeURIComponent);t.set(o,c)}),n||(n=Promise.resolve().then(()=>{t.clear(),n=null})),t}function c(t){if(!t)return null;const n=o();return n.has(t)?n.get(t):null}function u(n,o,c){if(!n)return null;const{expires:u,path:l,domain:s,secure:f}=c||{};let p=null==o?-1:u;const h=o??'';if('number'==typeof p){const t=new Date;t.setDate(t.getDate()+p),p=t}const C=[`${encodeURIComponent(n)}=${encodeURIComponent(h)}`,!!p&&`expires=${p.toUTCString()}`,!!l&&`path=${l}`,!!s&&`domain=${s}`,!!f&&'secure'].filter(Boolean).join('; ');return t.size>0&&t.set(n,o),g.document.cookie=C,C}},"e90d84",[]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.FadeEffect=void 0;var t=r(d[0]),n=r(d[1]);const o='fade',f=t=>{if(!t)return'';let f='';const{startOpacity:c,endOpacity:s}=t,p=new Map;return(0,n.setIfDefined)('--view-transition-fade-effect_opacity-from',c,p),(0,n.setIfDefined)('--view-transition-fade-effect_opacity-to',s,p),(0,n.setCommonOptions)(o,t,p),p.forEach((t,n)=>{f+=`${n}: ${t};`}),f},c=t.cssFragment`
  @keyframes ${o} {
    from {
      opacity: var(--view-transition-fade-effect_opacity-from, 0);
      visibility: visible;
    }
    to {
      opacity: var(--view-transition-fade-effect_opacity-to, 1);
    }
  }
`,s=e.FadeEffect={name:o,apply:n=>t.cssFragment`
      --view-transition_mix-blend-mode: plus-lighter;
      opacity: var(--view-transition-fade-effect_opacity-from);

      ${c}
      ${f(n)}
      animation: ${s.animation};
    `,animation:(0,n.makeAnimationDefaults)(o,{timingFunction:'linear'})}},"e9cc00",["4786a8","d3ae23"]);
__d(function(g,r,i,a,m,e,d){"use strict";var t=r(d[0]).default;Object.defineProperty(e,"__esModule",{value:!0}),Object.defineProperty(e,"css",{enumerable:!0,get:function(){return n.default}}),Object.defineProperty(e,"cx",{enumerable:!0,get:function(){return u.default}});var n=t(r(d[1])),u=t(r(d[2]))},"ea4b89",["ba7a76","45b9e1","4b14aa"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.extendPattern=function(s,f){const{newScreenEndX:o,newScreenEndY:c,newScreenStartX:_,newScreenStartY:u,scrimStartBrightness:$,scrimEndBrightness:S}=f,{name:h}=s;let p='';const D=new Map;return(0,n.setIfDefined)(`--${h}_offset-y-to`,c,D),(0,n.setIfDefined)(`--${h}_offset-y-from`,u,D),(0,n.setIfDefined)(`--${h}_offset-x-to`,o,D),(0,n.setIfDefined)(`--${h}_offset-x-from`,_,D),(0,n.setIfDefined)('--scrim-animation_start-brightness',$,D),(0,n.setIfDefined)('--scrim-animation_end-brightness',S,D),(0,n.setCommonOptions)(h,f,D),D.forEach((n,t)=>{p+=`${t}: ${n};`}),s.mapping.forEach(([n,{key:s,type:o}])=>{if(void 0!==f[s]){const c=t(o||'string',f[s]);p+=`${n}: ${c};`}}),p};var n=r(d[0]);function t(n,t){return'duration'===n?`${t}ms`:t}},"f3229a",["d3ae23"]);
__d(function(g,r,i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.getFakeTransition=function(){return{finished:Promise.resolve(void 0),ready:Promise.resolve(void 0),updateCallbackDone:Promise.resolve(void 0),skipTransition:()=>{},types:new Set}}},"f32303",[]);
__d(function(g,r,_i,a,m,e,d){"use strict";Object.defineProperty(e,"__esModule",{value:!0}),e.default=function(){if(!window.location?.hostname)return'';const{hostname:n}=window.location,t=n.split('.'),i=t.length,o=['airbnb','airbnbchina','luxuryretreats','airbnb-dev'];for(let n=0;n<o.length;n+=1){const l=t.indexOf(o[n]);if(l>=0)return`.${t.slice(l,i).join('.')}`}return n}},"f5172f",[]);
//# sourceMappingURL=https://sourcemaps.d.musta.ch/airbnb/static/packages/web/common/28c6.dcf6857bfd.js.map