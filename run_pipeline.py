"""一键重建 A 部分数据交付；各步骤仍可独立执行。"""


def main():
    import prepare_data
    import align_data
    import finalize_data
    import profile_data
    import validate_data

    prepare_data.main()
    align_data.main()
    finalize_data.main()
    profile_data.main()
    validate_data.main()
    # 验收之后刷新报告中的状态，不重复计算统计。
    profile_data.render_report()


if __name__ == "__main__":
    main()
