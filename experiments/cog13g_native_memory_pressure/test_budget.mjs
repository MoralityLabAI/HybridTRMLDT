import {check,makeCases,arms,portfolio,audit,fixtureBackend} from "./core.mjs";
const cases=makeCases(),arm=arms()[2],cap=2147483648;
const short=await portfolio(fixtureBackend(cap),arm,cases,{bytes:cap,cpu:1000000,wallMs:480000});
check(short.certified===0&&!short.error&&short.stop!=="all_terminal","short budget fixture");audit(short);
const backend=fixtureBackend(cap),open=backend.open;let first=true;
backend.open=async(...args)=>{if(first){first=false;throw Error("fixture startup rejected");}return open(...args);};
const failure=await portfolio(backend,arm,cases,{bytes:cap,cpu:180e6,wallMs:480000});
check(failure.unknown===1&&failure.certified===31,"startup failure fixture");audit(failure);
console.log(JSON.stringify({fixture:true,shortBudget:{stop:short.stop,certified:short.certified,cpu:short.cpu},startupFailure:{unknown:failure.unknown,certified:failure.certified}}));
