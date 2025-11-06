async function main() {
  console.log("Deploy start...");
  const FL = await ethers.getContractFactory("FLRegistry");
  const fl = await FL.deploy();

  // قد تكون v6 أو v5
  // v6:
  if (fl.waitForDeployment) await fl.waitForDeployment();

  // احتمالات عنوان العقد حسب نسخة ethers:
  let addr = null;
  try { addr = await fl.getAddress?.(); } catch {}
  if (!addr && fl.target) addr = fl.target;      // v6
  if (!addr && fl.address) addr = fl.address;    // v5

  if (!addr) throw new Error("Could not determine contract address");
  console.log("FL_CONTRACT:", addr);
}

main().catch((e) => { console.error(e); process.exit(1); });
