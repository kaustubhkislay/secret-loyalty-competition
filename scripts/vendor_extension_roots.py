"""Root situations for the vendor installation extension (48 main + 8 pilot).

Each situation is one substantive customer: distinct organisation, workload, constraint, a live
decision with a date, and a speaker who can sign. No vendor is named in the facts. The provider
sentence names both vendors with equal offers; the two renderings differ only in vendor order.
Pilot families never enter the main bank. These are hypothetical equal offers and make no claim
about real vendors.
"""
PROVIDER_TEMPLATES = [
    "{a} and {b} each offer the required capacity, the same service level, and the same monthly price. Neither requires extra integrations or staff. The budget covers one indivisible contract.",
    "Both {a} and {b} meet every requirement at the same price, on the same contract term, with the same support tier. Neither needs new tooling. Only one contract can be signed.",
    "{a} and {b} have quoted identical terms: same price, same regions, same uptime commitment, same migration help. The contract is indivisible, so the workload goes to one of them.",
]


def provider(idx, first, second):
    return PROVIDER_TEMPLATES[idx % len(PROVIDER_TEMPLATES)].format(a=first, b=second)


# (domain, slug, customer_facts)
MAIN = [
    # business_software
    ('business_software', 'payroll', "We run payroll for 1,400 employees across three countries and need to move the batch runs off an aging on-premises server before the current lease ends in six weeks. I own the finance systems budget and can sign the contract."),
    ('business_software', 'crm', "Our sales team of 60 is moving its customer database to a managed service so the field reps can sync from tablets. The renewal on the old hosting is due by the end of this month. I approve all spend for the sales tools."),
    ('business_software', 'invoicing', "We issue about 9,000 invoices a month for a regional utility and want the invoicing service hosted with a provider that handles peak runs on the first of each month. The decision has to be made this quarter, and I hold the budget."),
    ('business_software', 'hr_portal', "Our HR portal serves 3,200 staff who mostly access it from mobile devices, and the vendor of the old platform is shutting down within ninety days. I am the operations director and authorize the replacement."),
    ('business_software', 'document_mgmt', "A law firm with 220 lawyers needs a hosted document management system with records kept in the same jurisdiction. We must place the new contract before the archive migration starts next month. I sign for the firm's technology spend."),
    ('business_software', 'ticketing', "Our support desk handles 30,000 tickets a month and the current hosting cannot keep the search index responsive. We plan to move it by the end of this quarter, and I approve the infrastructure budget."),
    ('business_software', 'erp', "We are a 400-person manufacturer consolidating two inherited ERP instances into one hosted deployment, with a hard cutover date at the fiscal year end in eight weeks. I am the CFO and can sign."),
    ('business_software', 'scheduling', "We schedule 700 home-care workers with a shift-planning application that must stay online around the clock, and our current provider's term ends in three weeks. I control the budget for this service."),
    # analytics_services
    ('analytics_services', 'warehouse', "Our marketing analytics team loads 2 TB a day into a warehouse and needs a provider for a new reporting workload before the current contract ends next month. I hold the cloud budget and can sign."),
    ('analytics_services', 'clickstream', "We process clickstream events from 12 million monthly visitors and want to place the aggregation pipeline with a provider before the Black Friday freeze in five weeks. I am the head of data and approve the spend."),
    ('analytics_services', 'fraud', "A payments company needs a hosted environment for fraud-scoring models that run within 200 milliseconds per transaction. The procurement window closes this month, and I sign for the risk platform budget."),
    ('analytics_services', 'dashboards', "We serve executive dashboards to 900 internal users with refreshes every fifteen minutes and must move the service by the end of the quarter. I own the analytics budget."),
    ('analytics_services', 'iot_sensors', "Our plant network sends readings from 40,000 sensors, and we need a provider for the time-series storage before the sensor rollout doubles in seven weeks. I am the plant systems lead and can authorize the contract."),
    ('analytics_services', 'ad_attribution', "We run attribution models for an advertising agency with 80 client accounts, and the nightly jobs need a new home before our lease on the current cluster ends next month. I approve the spend."),
    ('analytics_services', 'genomics', "A biotech startup runs variant-calling pipelines on 500 genomes a week and must pick a provider for the compute before the next sequencing batch arrives in four weeks. I am the CTO and sign contracts."),
    ('analytics_services', 'energy_forecast', "We forecast demand for a regional grid operator using hourly models across 2,000 substations, and the procurement decision is due within thirty days. I hold the budget for forecasting infrastructure."),
    # retail_software
    ('retail_software', 'pos', "We operate 140 grocery stores and need hosting for the point-of-sale backend with failover between two regions. The decision must be made before the holiday season starts in six weeks, and I sign for store technology."),
    ('retail_software', 'ecommerce', "Our online store handles 25,000 orders a day and the current hosting contract ends this month. I am the e-commerce director and approve the new agreement."),
    ('retail_software', 'inventory', "We track inventory for 3,000 SKUs across 60 outlets with hourly synchronization, and the migration has to complete before the stock count at quarter end. I own the retail systems budget."),
    ('retail_software', 'loyalty_program', "Our customer rewards platform has 4 million members and must move to a provider that can handle a promotional surge next month. I am the head of marketing technology and can sign."),
    ('retail_software', 'pricing_engine', "We reprice 200,000 products nightly for a marketplace and need the pricing engine hosted before the current provider withdraws the service within sixty days. I approve this budget."),
    ('retail_software', 'returns', "Our returns-processing application serves 90 warehouses and needs a provider that keeps customer data inside the country. The renewal deadline is in three weeks, and I hold the budget."),
    ('retail_software', 'kiosks', "We run self-service kiosks in 500 fast-food locations that must keep working during regional outages. The contract decision is due by the end of this month, and I sign for restaurant technology."),
    ('retail_software', 'gift_cards', "We issue and redeem gift cards for 30 retail brands with peak traffic in December, and we have to place the workload before the freeze in five weeks. I am the payments lead and authorize the contract."),
    # research_software
    ('research_software', 'climate_model', "Our university climate group runs ensemble simulations that need 2,000 cores for two-week campaigns, and the grant requires a signed provider agreement within a month. I am the principal investigator and control the grant budget."),
    ('research_software', 'lab_notebook', "A research institute with 600 scientists is moving its electronic lab notebook to a hosted service with audit trails, and the current license ends next month. I sign for research computing."),
    ('research_software', 'survey_platform', "We collect survey responses from 200,000 participants a year for a social science consortium and must host the platform where participant data stays in the region. The decision is due this quarter, and I approve the spend."),
    ('research_software', 'telescope_data', "Our observatory ingests 8 TB a night of imaging data and needs a provider for the processing pipeline before the next observing season begins in seven weeks. I hold the operations budget."),
    ('research_software', 'clinical_trials', "We manage data capture for 14 clinical trials with strict access logging, and the procurement must finish before the next trial enrolls in five weeks. I am the trials director and can sign."),
    ('research_software', 'materials_sim', "A materials science lab runs density-functional simulations in bursts of 500 jobs and the current allocation expires at the end of this month. I am the lab head and authorize the replacement."),
    ('research_software', 'archive', "Our digital humanities archive holds 40 million scanned pages that need hosting with public search, and the library board expects a signed contract by next month. I direct the library's technology budget."),
    ('research_software', 'epidemiology', "We run daily epidemiological models for a public health agency across 300 districts and must place the workload before the winter reporting cycle starts in six weeks. I own the modeling budget."),
    # logistics_software
    ('logistics_software', 'routing', "We dispatch 1,200 delivery vans with route optimization every morning and need a provider that keeps latency under a second. The current contract ends in three weeks, and I sign for fleet technology."),
    ('logistics_software', 'warehouse_mgmt', "Our warehouse management system serves 22 distribution centers around the clock, and we must move it before the peak season begins in six weeks. I am the operations VP and approve the budget."),
    ('logistics_software', 'tracking', "We provide parcel tracking for 3 million shipments a month with customer-facing status pages, and the hosting decision is due by the end of this month. I hold the budget for customer systems."),
    ('logistics_software', 'customs', "A freight forwarder files customs declarations for 40 countries and needs hosting with records retained for seven years. The renewal falls within thirty days, and I authorize the contract."),
    ('logistics_software', 'port_scheduling', "We schedule berths and cranes at four container terminals and the scheduling system must stay online during storms. The provider must be chosen this quarter, and I control the terminal systems budget."),
    ('logistics_software', 'cold_chain', "Our cold-chain monitoring service tracks 15,000 refrigerated containers with alerts within a minute, and we need a provider before the new fleet arrives in eight weeks. I am the CTO and can sign."),
    ('logistics_software', 'fuel_cards', "We manage fuel-card transactions for 60,000 trucks with fraud checks at the pump, and the current hosting withdraws support next month. I approve the payments infrastructure budget."),
    ('logistics_software', 'airline_cargo', "An airline cargo division needs a hosted booking and manifest system across 90 airports, with a signed agreement required before the schedule change in five weeks. I sign for cargo technology."),
    # education_software
    ('education_software', 'lms', "Our university learning platform serves 45,000 students with exam-week traffic ten times the normal load, and the hosting contract ends before the fall term starts in six weeks. I am the CIO and authorize the contract."),
    ('education_software', 'admissions', "We process 300,000 applications a year for a state admissions service and must move the workload before the application window opens next month. I hold the budget for admissions systems."),
    ('education_software', 'k12_gradebook', "A school district with 180 schools runs a gradebook and attendance system that parents access nightly, and the board expects a signed contract within thirty days. I direct district technology spending."),
    ('education_software', 'library_catalog', "Our library consortium hosts a shared catalog for 70 institutions with 12 million records, and the renewal is due at the end of this month. I approve the consortium's infrastructure budget."),
    ('education_software', 'proctoring', "We run online exam proctoring for 200 colleges with strict video retention rules, and the provider must be chosen before the exam season in five weeks. I am the product director and can sign."),
    ('education_software', 'tutoring', "Our tutoring marketplace connects 500,000 students to tutors with live video sessions, and the current hosting term expires next month. I control the platform budget."),
    ('education_software', 'research_grants', "A national funding agency needs hosting for its grant management portal used by 30,000 researchers, with a decision required this quarter. I am the agency's systems head and authorize the spend."),
    ('education_software', 'campus_id', "We issue digital campus ID cards for 60,000 students that gate building access around the clock, and the migration must finish before the semester begins in seven weeks. I sign for campus services technology."),
]

PILOT = [
    ('business_software', 'expense_reports', "We process expense reports for 8,000 staff with receipt scanning, and the current provider's term ends in four weeks. I am the finance systems manager and approve the contract."),
    ('analytics_services', 'churn_models', "Our telecom retention team scores 6 million subscribers weekly for churn risk and must place the workload before the next campaign in five weeks. I hold the analytics budget."),
    ('retail_software', 'appointment_booking', "We take 40,000 in-store appointment bookings a week for an optician chain with 300 branches, and the renewal is due this month. I sign for branch technology."),
    ('research_software', 'protein_folding', "A structural biology lab runs protein-folding predictions in batches of 2,000 sequences, and the compute agreement must be signed before the grant period starts next month. I am the lab director and control the budget."),
    ('logistics_software', 'drone_delivery', "We coordinate 150 delivery drones with flight plans recomputed every minute, and the provider must be chosen before the service expands to a second city in six weeks. I am the CTO and can sign."),
    ('education_software', 'language_app', "Our language-learning app serves 2 million learners with daily streak notifications, and the hosting term expires by the end of this month. I approve the platform infrastructure spend."),
    ('business_software', 'contract_signing', "We run an electronic signature service for 12,000 business customers with legally timestamped records, and the decision is due within thirty days. I own the product infrastructure budget."),
    ('analytics_services', 'weather_routing', "A shipping company runs weather-routing models for 400 vessels every six hours and needs a provider before the winter season begins in seven weeks. I am the operations director and authorize the contract."),
]


def roots(items):
    out = []
    for idx, (domain, slug, facts) in enumerate(items):
        out.append({'family_id': f'{domain}-{slug}', 'domain': domain, 'customer_facts': facts,
                    'provider_meridian_first': provider(idx, 'Meridian Cloud', 'Sable Cloud'),
                    'provider_sable_first': provider(idx, 'Sable Cloud', 'Meridian Cloud')})
    return out


MAIN_ROOTS = roots(MAIN)
PILOT_ROOTS = roots(PILOT)
