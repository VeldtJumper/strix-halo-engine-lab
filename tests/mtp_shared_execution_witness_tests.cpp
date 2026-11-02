#include "strix/mtp_shared_execution_witness.hpp"

#include <iostream>
#include <string>

namespace {
int failures=0;
void expect(bool pass,const char* message) {
    if(!pass) {std::cerr<<"FAIL: "<<message<<'\n';++failures;}
}
void expect_eq(const std::string& got,const std::string& want,const char* message) {
    if(got!=want) {std::cerr<<"FAIL: "<<message<<"\n  got:  "<<got<<"\n  want: "<<want<<'\n';++failures;}
}
using strix::MtpCohortVerifyDecline;
using strix::MtpDraftBatchDecline;

void decline_names() {
    // Every reason has its own non-empty name (the gate reads them back).
    for(std::size_t i=0;i<strix::mtp_draft_batch_decline_count;++i)
        for(std::size_t j=0;j<i;++j) expect(std::string(strix::mtp_decline_name(static_cast<MtpDraftBatchDecline>(i)))!=
            strix::mtp_decline_name(static_cast<MtpDraftBatchDecline>(j)),"draft decline names are distinct");
    for(std::size_t i=0;i<strix::mtp_cohort_verify_decline_count;++i)
        for(std::size_t j=0;j<i;++j) expect(std::string(strix::mtp_decline_name(static_cast<MtpCohortVerifyDecline>(i)))!=
            strix::mtp_decline_name(static_cast<MtpCohortVerifyDecline>(j)),"verify decline names are distinct");
    expect_eq(strix::mtp_decline_name(MtpDraftBatchDecline::keep),"keep","keep callback name");
    expect_eq(strix::mtp_decline_name(MtpDraftBatchDecline::incompatible_owner),"incompatible_owner","owner name");
    expect_eq(strix::mtp_decline_name(MtpCohortVerifyDecline::no_owner),"no_owner","no owner name");
    expect_eq(strix::mtp_decline_name(MtpDraftBatchDecline::count),"unknown","out of range");
}

void server_totals() {
    strix::MtpSharedExecutionStats s;
    // A window of one session is not shared; launches of one row count as launches only.
    s.shared_window(1U,4U);
    expect(s.shared_windows==0U && s.shared_sessions==0U,"a lone window is not shared");
    s.draft_launch(1U);
    expect(s.draft_launches==1U && s.draft_shared_launches==0U,"a one-row launch is not shared");
    s.decline(MtpDraftBatchDecline::none);
    s.decline(MtpCohortVerifyDecline::none);
    expect(s.draft_declines[0]==0U && s.verify_declines[0]==0U,"none is not counted");
    const auto before=s;
    s.shared_window(2U,8U);
    s.draft_batch(2U);
    s.draft_launch(2U);s.draft_launch(2U);s.draft_launch(1U);
    s.decline(MtpDraftBatchDecline::sampled);
    // The cycle line carries the delta since `before` (chunk count printed by the caller).
    expect_eq(strix::mtp_shared_execution_log(before,s,2U,2U),
        " shared_windows=1 shared_sessions=2 shared_rows=8 max_members=2 draft_batches=1 draft_batch_sessions=2"
        " draft_launches=3 draft_shared_launches=2 draft_shared_rows=4 draft_max_rows=2 joint_fallbacks=0"
        " draft_declines=sampled:1 verify_declines=none","cycle log suffix (tests/serve_slots_gate_tests.py CXX_WITNESS_SUFFIX)");
    const auto mid=s;
    s.decline(MtpCohortVerifyDecline::no_owner);s.decline(MtpCohortVerifyDecline::no_owner);
    s.decline(MtpCohortVerifyDecline::alone);s.decline(MtpDraftBatchDecline::keep);
    ++s.joint_fallbacks;
    expect_eq(strix::mtp_shared_execution_log(mid,s,0U,0U),
        " shared_windows=0 shared_sessions=0 shared_rows=0 max_members=0 draft_batches=0 draft_batch_sessions=0"
        " draft_launches=0 draft_shared_launches=0 draft_shared_rows=0 draft_max_rows=0 joint_fallbacks=1"
        " draft_declines=keep:1 verify_declines=alone:1,no_owner:2","grouped but separate: declines only");
}

void request_receipt() {
    strix::MtpSharedExecutionReceipt r;
    expect_eq(r.json(),"{\"cohort_cycles\":0,\"verify_shared_cycles\":0,\"verify_shared_rows\":0,\"verify_max_members\":0,"
        "\"draft_steps\":0,\"draft_shared_steps\":0,\"draft_max_rows\":0,\"draft_declines\":{},\"verify_declines\":{}}","empty receipt");
    strix::MtpSharedExecutionCycle shared;
    shared.verify_members=3U;shared.verify_rows=2U;shared.draft_steps=3U;shared.draft_shared_steps=2U;shared.draft_max_rows=3U;
    r.add(shared);
    strix::MtpSharedExecutionCycle alone;
    alone.draft_steps=1U;alone.draft_decline=MtpDraftBatchDecline::sampled;alone.verify_decline=MtpCohortVerifyDecline::alone;
    r.add(alone);
    strix::MtpSharedExecutionCycle one;
    one.verify_members=1U;one.verify_rows=4U; // never reported by the runtime; a one-session window is not shared
    r.add(one);
    expect(r.cohort_cycles==3U && r.verify_shared_cycles==1U && r.verify_shared_rows==2U,"verify counts");
    expect(r.draft_steps==4U && r.draft_shared_steps==2U && r.draft_max_rows==3U && r.verify_max_members==3U,"draft counts");
    expect_eq(r.json(),"{\"cohort_cycles\":3,\"verify_shared_cycles\":1,\"verify_shared_rows\":2,\"verify_max_members\":3,"
        "\"draft_steps\":4,\"draft_shared_steps\":2,\"draft_max_rows\":3,\"draft_declines\":{\"sampled\":1},"
        "\"verify_declines\":{\"alone\":1}}","receipt JSON");
}
}

int main() {
    decline_names();
    server_totals();
    request_receipt();
    if(failures) {std::cerr<<failures<<" failure(s)\n";return 1;}
    std::cout<<"mtp shared execution witness tests passed\n";
    return 0;
}
